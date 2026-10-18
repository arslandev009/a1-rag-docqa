import { handle } from "../../lib/gateway.js";

export const onRequest = (context) => handle("session", context.request, context.env);
