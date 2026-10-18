import { handle } from "../../lib/gateway.js";

export const onRequest = (context) => handle("chat", context.request, context.env);
