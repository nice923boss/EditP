// Images dropped on the window or pasted with Ctrl+V go to /api/upload (saved under Pictures\EditP, then opened).
const TYPES = /^image\/(png|jpeg|webp|bmp|tiff|gif)$/;

const firstImage = (files) => [...(files || [])].find((f) => TYPES.test(f.type));

export function setupUpload({ target, onFile, onRejected }) {
  let depth = 0; // dragenter/dragleave fire for every child element
  const hasFiles = (ev) => [...(ev.dataTransfer?.types || [])].includes("Files");

  window.addEventListener("dragenter", (ev) => {
    if (!hasFiles(ev)) return;
    depth += 1;
    target.classList.add("dropping");
  });
  window.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) target.classList.remove("dropping");
  });
  window.addEventListener("dragover", (ev) => { if (hasFiles(ev)) ev.preventDefault(); });
  window.addEventListener("drop", (ev) => {
    if (!hasFiles(ev)) return;
    ev.preventDefault(); // otherwise the browser leaves the app to show the file
    depth = 0;
    target.classList.remove("dropping");
    const file = firstImage(ev.dataTransfer.files);
    if (file) onFile(file, file.name); else onRejected("拖進來的不是 PNG、JPG、WebP、BMP、TIFF 或 GIF 圖片");
  });
  window.addEventListener("paste", (ev) => {
    if (document.querySelector("dialog[open]")) return;
    const file = firstImage(ev.clipboardData?.files);
    if (!file) return; // plain text paste stays with the text field
    ev.preventDefault();
    onFile(file, ""); // the clipboard calls every image image.png: let the server name it by time
  });
}

export async function uploadFile(file, name) {
  const res = await fetch(`/api/upload?name=${encodeURIComponent(name)}`, {
    method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
  return data;
}
