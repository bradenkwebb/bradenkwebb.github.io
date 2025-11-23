(function () {
  const THEME_KEY = "theme";
  const ICON = {
    light: "☀️",
    dark: "🌙",
    system: "🖥️",
  };
  const states = ["light", "dark", "system"];

  function prefersDark() {
    return (
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
    );
  }

  function applyTheme(theme, persist = true) {
    const root = document.documentElement;
    let effective = theme;
    if (!theme || theme === "system") {
      effective = prefersDark() ? "dark" : "light";
    }
    if (effective === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.setAttribute("aria-pressed", effective === "dark");
      btn.title = "Theme: " + theme;
      const iconEl = document.getElementById("theme-toggle-icon");
      if (iconEl) iconEl.textContent = ICON[theme] || ICON.system;
    }
    if (persist) {
      try {
        localStorage.setItem(THEME_KEY, theme);
      } catch (e) {}
    }
    // Debug/log
    try {
      console.debug("applyTheme:", { theme, effective });
    } catch (e) {}

    // Also set CSS custom properties and fallback inline colors so
    // pages that still have compile-time colors get visibly updated.
    const VARS = {
      light: {
        "--color-text": "#111111",
        "--color-bg": "#fdfdfd",
        "--color-brand": "#2a7ae2",
        "--color-code-bg": "#eef",
      },
      dark: {
        "--color-text": "#f5f5f5",
        "--color-bg": "#0b0b0b",
        "--color-brand": "#2a7ae2",
        "--color-code-bg": "#232323",
      },
    };

    // Apply CSS variables on root
    const vars = VARS[effective] || VARS.light;
    Object.keys(vars).forEach((k) => {
      try {
        root.style.setProperty(k, vars[k]);
      } catch (e) {}
    });

    // Fallback: set body inline color/background explicitly
    try {
      document.body.style.backgroundColor = vars["--color-bg"];
      document.body.style.color = vars["--color-text"];
    } catch (e) {}
  }

  function loadTheme() {
    try {
      return localStorage.getItem(THEME_KEY) || "system";
    } catch (e) {
      return "system";
    }
  }

  function cycleTheme() {
    const current = loadTheme();
    const idx = states.indexOf(current);
    const next = states[(idx + 1) % states.length];
    applyTheme(next, true);
  }

  // Listen for OS theme changes when in 'system' mode
  let mql = null;
  function systemChangeHandler() {
    if (loadTheme() === "system") {
      applyTheme("system", false);
    }
  }
  function setupSystemListener() {
    if (mql) {
      try {
        mql.removeEventListener("change", systemChangeHandler);
      } catch (e) {}
    }
    if (window.matchMedia) {
      mql = window.matchMedia("(prefers-color-scheme: dark)");
      if (mql.addEventListener) {
        mql.addEventListener("change", systemChangeHandler);
      } else if (mql.addListener) {
        mql.addListener(systemChangeHandler);
      }
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", function (e) {
      cycleTheme();
    });
    setupSystemListener();
    applyTheme(loadTheme(), false);
  });

  // Expose for debugging
  window.__theme = {
    apply: applyTheme,
    set: function (t) {
      applyTheme(t, true);
    },
    get: loadTheme,
  };
})();
