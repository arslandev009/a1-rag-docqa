import { handle } from "../../lib/gateway.js";

export const onRequest = (context) => handle("sources", context.request, context.env);
