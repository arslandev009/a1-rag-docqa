import { handle } from "../../lib/gateway.js";

export const onRequest = (context) => handle("warmup", context.request, context.env);
