// Undo/redo for the edit list. Each step is a snapshot of the list together with the render shown for it, so
// undoing「全部套用」brings the previous picture back, and undoing again reverts the list edits before it.
const LIMIT = 200;
const MERGE_MS = 1500; // keystrokes in one field closer together than this are one step

export function createHistory() {
  let steps = [];
  let index = -1;
  let canMerge = false; // false right after undo/redo, so a new edit never folds into a step the user went back to

  return {
    reset(snap) {
      steps = [{ ...snap, time: Date.now() }];
      index = 0;
      canMerge = false;
    },
    // a change happened: drop the redo branch, then add a step (or extend the last one when tagged the same)
    push(snap) {
      const now = Date.now();
      const top = steps[index];
      steps = steps.slice(0, index + 1);
      if (canMerge && top && snap.tag && top.tag === snap.tag && now - top.time < MERGE_MS) {
        steps[index] = { ...snap, label: top.label, time: now };
        return;
      }
      steps.push({ ...snap, time: now });
      if (steps.length > LIMIT) steps.shift();
      index = steps.length - 1;
      canMerge = true;
    },
    // replace the current step's content, keeping its label (an automatic re-render after a style tweak)
    amend(snap) {
      if (index >= 0) steps[index] = { ...snap, label: steps[index].label, tag: steps[index].tag, time: Date.now() };
    },
    undo() {
      if (index <= 0) return null;
      canMerge = false;
      return { undone: steps[index], step: steps[--index] };
    },
    redo() {
      if (index >= steps.length - 1) return null;
      canMerge = false;
      return { step: steps[++index] };
    },
    nextUndo: () => (index > 0 ? steps[index] : null),
    nextRedo: () => (index < steps.length - 1 ? steps[index + 1] : null),
    // rewrite every stored step (the AI read corrects the original text of items in all of them)
    patch(fn) {
      steps = steps.map(fn);
    },
  };
}
