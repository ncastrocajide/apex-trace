// Entry point: the two views and the tabs that switch between them.

import { initAnalysis } from "./analysis.js";
import { initLive } from "./live.js";

const tabs = [...document.querySelectorAll('[role="tab"]')];

function show(name) {
  for (const tab of tabs) {
    const selected = tab.id === `tab-${name}`;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
  }
  history.replaceState(null, "", `#${name}`);
}

for (const tab of tabs) {
  tab.addEventListener("click", () => show(tab.id.slice(4)));
  // Arrow keys move between tabs, as the tab pattern expects.
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const next = tabs[(tabs.indexOf(tab) + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    next.focus();
    show(next.id.slice(4));
  });
}

initAnalysis();
initLive();
show(location.hash === "#live" ? "live" : "analysis");
