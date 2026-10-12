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

CONTINUE = "CONTINUE"
STOP = "STOP"


def make_request(job, step, checkpoint="", train_loss=None):
    return {"type": EVAL_REQUEST, "job": job, "step": step,
            "checkpoint": checkpoint, "train_loss": train_loss}


def make_verdict(step, decision, reason=""):
    assert decision in (CONTINUE, STOP), decision
    return {"type": VERDICT, "step": step, "decision": decision,
            "reason": reason}


class EvalGate:
    """Trainer-side (Kaggle) sync client. CPU-only, no GPU touched.

    proxy: optional (host, port) SOCKS5 for userspace-tailscale Kaggle
    (e.g. ("localhost", 1055)). Direct connection when None.
    """

    def __init__(self, url, every=2, timeout=30, proxy=None):
        self.url = url
        self.every = every
        self.timeout = timeout
        self.proxy = proxy
        self._pending_stop = False

    def due(self, step):
        return step > 0 and step % self.every == 0

    def _connect(self):
        import websocket
        if self.proxy:
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
                                               train_loss)))
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
            if self.proxy:
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


def run_server(host, port, decide, timeout_note=""):
    """Laptop-side asyncio server. decide(step:int, payload:dict)
    -> (decision, reason). Runs until cancelled. CPU-only."""
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
                    await ws.send(json.dumps({"type": ACK,
                                              "step": msg.get("step", -1)}))
                else:
                    await ws.send(json.dumps({"type": ERROR,
                                              "detail": f"unknown type {mtype}"}))
        except Exception:
            pass

    async def _main():
        async with await websockets.serve(handler, host, port):
            await asyncio.Future()

    asyncio.run(_main())
