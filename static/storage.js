// The batch list of each image survives a page reload or a browser restart (kept in this browser only).
const PREFIX = "editp:list:";
const KEEP = 20; // images remembered; the oldest list is dropped beyond that

function entries() {
  const out = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k.startsWith(PREFIX)) out.push([k, JSON.parse(localStorage.getItem(k)).at || 0]);
    }
  } catch {
    // storage blocked or an entry is broken: nothing to trim
  }
  return out;
}

export function loadList(imageKey) {
  try {
    const saved = JSON.parse(localStorage.getItem(PREFIX + imageKey));
    return Array.isArray(saved?.items) && saved.items.length ? saved.items : null;
  } catch {
    return null;
  }
}

export function saveList(imageKey, items) {
  if (!imageKey) return;
  try {
    if (!items.length) return localStorage.removeItem(PREFIX + imageKey);
    localStorage.setItem(PREFIX + imageKey, JSON.stringify({ at: Date.now(), items }));
    const all = entries().sort((a, b) => b[1] - a[1]);
    for (const [k] of all.slice(KEEP)) localStorage.removeItem(k);
  } catch {
    // storage full or blocked: the list still works, it is just not kept across reloads
  }
}
