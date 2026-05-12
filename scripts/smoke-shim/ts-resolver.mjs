// Tiny Node resolver hook: when a specifier ending in `.js` resolves to a
// missing file but the same path with `.ts` exists, rewrite the specifier.
// The TypeScript source in this repo follows the NodeNext convention of
// writing `.js` extensions in import paths; Node's --experimental-strip-types
// does not rewrite those automatically, so we do it here.
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

export async function resolve(specifier, context, nextResolve) {
  if (typeof specifier === "string" && (specifier.startsWith("./") || specifier.startsWith("../")) && specifier.endsWith(".js")) {
    try {
      const parentUrl = context.parentURL ? new URL(context.parentURL) : undefined;
      if (parentUrl) {
        const candidateUrl = new URL(specifier, parentUrl);
        const candidatePath = fileURLToPath(candidateUrl);
        if (!existsSync(candidatePath)) {
          const tsPath = candidatePath.replace(/\.js$/, ".ts");
          if (existsSync(tsPath)) {
            return nextResolve(pathToFileURL(tsPath).href, context);
          }
        }
      }
    } catch {
      /* fall through to default resolution */
    }
  }
  return nextResolve(specifier, context);
}
