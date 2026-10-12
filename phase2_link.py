"""Phase-2 control plane: async eval verdicts over websocket (Tailscale on GPU day).

Protocol (JSON, one message per websocket frame):
  trainer -> laptop  {"type": "EVAL_REQUEST", "job": str, "step": int,
                      "checkpoint": str, "train_loss": float}
  laptop  -> trainer {"type": "VERDICT", "step": int, "decision": "CONTINUE" | "STOP",
                      "reason": str}
  trainer -> laptop  {"type": "CHECKPOINT_DONE", "step": int, "stopped": bool}
  laptop  -> trainer {"type": "ACK", "step": int}
  laptop  -> trainer {"type": "ERROR", "detail": str}   (malformed input)

Semantics: trainer notifies after every `every` checkpoints and blocks up to
`timeout` for a verdict. ANY failure/timeout -> CONTINUE (fail-open: training
never hangs on eval). On STOP the trainer finishes its current checkpoint
(X+1), reports CHECKPOINT_DONE(stopped=True), then exits.

Laptop side runs the server and plugs the real evaluator into `decide`.
Kaggle side uses EvalGate from the sync training loop (websocket-client).

GPU-day network (verified 2026-10-12, tailscaled up):
  LAPTOP_TAIL_IP = "100.91.220.13"   (windows-lappy)
  WS_PORT = 8765                     (bind 0.0.0.0 on laptop; allow inbound
                                      Windows firewall on Tailscale iface)
  Kaggle EvalGate url = "ws://100.91.220.13:8765"
  Kaggle Neo4j url    = "bolt://100.91.220.13:7687" (or http://...:7474/...)
  Kaggle joins tailnet via auth key: tailscale up --auth-key=$TS_KEY
"""
import json

EVAL_REQUEST = "EVAL_REQUEST"
VERDICT = "VERDICT"
CHECKPOINT_DONE = "CHECKPOINT_DONE"
ACK = "ACK"
ERROR = "ERROR"
ROLLING_ROWS = "ROLLING_ROWS"
BASE_EVAL_REQUEST = "BASE_EVAL_REQUEST"
BASE_EVAL_ACK = "BASE_EVAL_ACK"

CONTINUE = "CONTINUE"
STOP = "STOP"


def make_request(job, step, checkpoint="", train_loss=None, auth=None):
    return {"type": EVAL_REQUEST, "job": job, "step": step,
            "checkpoint": checkpoint, "train_loss": train_loss,
            "auth": auth}


def make_verdict(step, decision, reason=""):
    assert decision in (CONTINUE, STOP), decision
    return {"type": VERDICT, "step": step, "decision": decision,
            "reason": reason}


class EvalGate:
    """Trainer-side (Kaggle) sync client. CPU-only, no GPU touched.

    proxy: optional (host, port) SOCKS5 for userspace-tailscale Kaggle
    (e.g. ("localhost", 1055)). Direct connection when None.
    """

    def __init__(self, url, every=2, timeout=30, proxy=None, auth=None):
        self.url = url
        self.every = every
        self.timeout = timeout
        self.proxy = proxy
        self.auth = auth
        self._pending_stop = False

    def due(self, step):
        return step > 0 and step % self.every == 0

    def _connect(self):
        import websocket
        # Public wss:// always goes direct (Kaggle has internet); only
        # tailnet ws:// rides the userspace SOCKS proxy. Proxying TLS
        # through SOCKS breaks the handshake (SSLEOF).
        if self.proxy and self.url.startswith("ws://"):
            host, port = self.proxy
            return websocket.create_connection(
                self.url, timeout=self.timeout,
                http_proxy_host=host, http_proxy_port=port,
                proxy_type="socks5h")
        return websocket.create_connection(self.url, timeout=self.timeout)

    def notify(self, step, checkpoint="", train_loss=None, job="phase2"):
        """Ask laptop for a verdict. Always returns CONTINUE or STOP;
        never raises (fail-open to CONTINUE)."""
        import websocket
        try:
            ws = self._connect()
            try:
                ws.send(json.dumps(make_request(job, step, checkpoint,
                                               train_loss, auth=self.auth)))
                ws.settimeout(self.timeout)
                raw = ws.recv()
            finally:
                ws.close()
            msg = json.loads(raw)
            if msg.get("type") == VERDICT and msg.get("step") == step:
                decision = msg.get("decision", CONTINUE)
                if decision == STOP:
                    self._pending_stop = True
                return decision if decision in (CONTINUE, STOP) else CONTINUE
            return CONTINUE
        except Exception:
            return CONTINUE

    @property
    def stop_requested(self):
        return self._pending_stop

    def report_done(self, step, stopped=False):
        """Best-effort CHECKPOINT_DONE; never raises."""
        import websocket
        try:
            if self.proxy and self.url.startswith("ws://"):
                host, port = self.proxy
                ws = websocket.create_connection(
                    self.url, timeout=10, http_proxy_host=host,
                    http_proxy_port=port, proxy_type="socks5h")
            else:
                ws = websocket.create_connection(self.url, timeout=10)
            try:
                ws.send(json.dumps({"type": CHECKPOINT_DONE, "step": step,
                                    "stopped": bool(stopped)}))
            finally:
                ws.close()
        except Exception:
            pass


class RollStream:
    """Trainer-side buffered sender for RL rolling rows. Fail-open: never
    raises; drops rows when unreachable. Flushes every `flush_every` rows
    over a fresh websocket (no long-lived connection to babysit)."""

    def __init__(self, url, auth=None, proxy=None, flush_every=50):
        self.url = url
        self.auth = auth
        self.proxy = proxy
        self.flush_every = max(1, flush_every)
        self.buf = []

    def add(self, rows):
        try:
            self.buf.extend(rows)
            if len(self.buf) >= self.flush_every:
                self.flush()
        except Exception:
            pass

    def flush(self):
        import websocket
        import json as _json
        rows, self.buf = self.buf, []
        if not rows or not self.url:
            return
        try:
            kw = dict(timeout=20)
            if self.proxy and self.url.startswith("ws://"):
                kw.update(http_proxy_host=self.proxy[0],
                          http_proxy_port=self.proxy[1],
                          proxy_type="socks5h")
            ws = websocket.create_connection(self.url, **kw)
            try:
                ws.send(_json.dumps({"type": ROLLING_ROWS, "step": rows[-1].get("step", -1),
                                     "rows": rows, "auth": self.auth}))
            finally:
                ws.close()
        except Exception:
            pass


class RollingCollector:
    """Laptop-side sink: buckets rows into per-checkpoint-window jsonl files.

    With hf_repo set, each closed window uploads to
    `<hf_prefix>/rolling_ckpt_<base>.jsonl` (upload on rollover and on
   1632 finalize()). Uploads never raise (fail-open, local files are source
    of truth).
    """

    def __init__(self, out_dir, window=500, hf_repo="", hf_token="",
                 hf_prefix="PHASE_2_KAGGLE/rolling"):
        import pathlib
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.hf_repo = hf_repo
        self.hf_token = hf_token
        self.hf_prefix = hf_prefix.strip("/")
        self._cur_base = None

    def ingest(self, step, rows):
        import json as _json
        base = (int(step) // self.window) * self.window
        if self._cur_base is not None and base != self._cur_base:
            self.upload_window(self._cur_base)
        self._cur_base = base
        path = self.out_dir / f"rolling_ckpt_{base}.jsonl"
        with open(path, "a") as f:
            for r in rows:
                f.write(_json.dumps(r) + "\n")
        return str(path)

    def files(self):
        return sorted(self.out_dir.glob("rolling_ckpt_*.jsonl"))

    def upload_window(self, base):
        if not self.hf_repo:
            return None
        try:
            from huggingface_hub import upload_file
            path = self.out_dir / f"rolling_ckpt_{base}.jsonl"
            if not path.exists():
                return None
            where = upload_file(path_or_fileobj=str(path),
                                path_in_repo=f"{self.hf_prefix}/rolling_ckpt_{base}.jsonl",
                                repo_id=self.hf_repo, token=self.hf_token or None)
            print(f"[rolling] uploaded window {base} -> {where}", flush=True)
            return where
        except Exception as e:
            print(f"[rolling] upload failed window {base}: {e}", flush=True)
            return None

    def finalize(self, step):
        """Upload the in-progress window (call on CHECKPOINT_DONE)."""
        base = (int(step) // self.window) * self.window
        self.upload_window(base)
        self._cur_base = base


def run_server(host, port, decide, auth=None, rolling_dir=None,
               on_base_eval=None, rolling_hf_repo="", rolling_hf_token=""):
    """Laptop-side asyncio server. decide(step:int, payload:dict)
    -> (decision, reason). Runs until cancelled. CPU-only.
    auth: when set, EVAL_REQUEST must carry the same token or gets ERROR
    (required when exposed over public funnel)."""
    import asyncio
    import websockets

    async def handler(ws):
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    await ws.send(json.dumps({"type": ERROR,
                                              "detail": "bad json"}))
                    continue
                mtype = msg.get("type")
                if mtype == EVAL_REQUEST:
                    if auth is not None and msg.get("auth") != auth:
                        await ws.send(json.dumps({"type": ERROR,
                                                  "detail": "bad auth"}))
                        continue
                    step = int(msg.get("step", -1))
                    try:
                        decision, reason = decide(step, msg)
                    except Exception as e:
                        decision, reason = CONTINUE, f"decide failed: {e}"
                    if decision not in (CONTINUE, STOP):
                        decision = CONTINUE
                    await ws.send(json.dumps(make_verdict(step, decision,
                                                         str(reason))))
                elif mtype == CHECKPOINT_DONE:
                    if _collector is not None:
                        _collector.finalize(msg.get("step", -1))
                    await ws.send(json.dumps({"type": ACK,
                                              "step": msg.get("step", -1)}))
                elif mtype == ROLLING_ROWS:
                    if auth is not None and msg.get("auth") != auth:
                        await ws.send(json.dumps({"type": ERROR,
                                                  "detail": "bad auth"}))
                        continue
                    rows = msg.get("rows", [])
                    if rolling_dir is not None:
                        _collector.ingest(msg.get("step", -1), rows)
                    await ws.send(json.dumps({"type": ACK,
                                              "step": msg.get("step", -1),
                                              "rows": len(rows)}))
                elif mtype == BASE_EVAL_REQUEST:
                    if auth is not None and msg.get("auth") != auth:
                        await ws.send(json.dumps({"type": ERROR,
                                                  "detail": "bad auth"}))
                        continue
                    note = ("queued (stub)" if on_base_eval is None
                            else on_base_eval(int(msg.get("step", 0)), msg))
                    await ws.send(json.dumps({"type": BASE_EVAL_ACK,
                                              "step": msg.get("step", 0),
                                              "note": str(note)}))
                else:
                    await ws.send(json.dumps({"type": ERROR,
                                              "detail": f"unknown type {mtype}"}))
        except Exception:
            pass

    _collector = (RollingCollector(rolling_dir, hf_repo=rolling_hf_repo,
                                     hf_token=rolling_hf_token)
                    if rolling_dir else None)

    async def _main():
        async with await websockets.serve(handler, host, port):
            await asyncio.Future()

    asyncio.run(_main())
