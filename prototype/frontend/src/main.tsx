import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { initDevLog } from "./devLog";
import { bootstrapPrefs } from "./api/prefs";
import "./styles.css";

initDevLog();

// Pull shared prefs (recents + UI config) from the backend into localStorage
// before the first render, so config is the same in any browser regardless of
// origin or port. Renders even if the server is offline.
bootstrapPrefs().finally(() => {
  createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
});

