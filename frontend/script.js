(() => {
  const state = {
    charts: {},
    options: null,
    chartsPayload: null,
    overview: null,
    modelDetails: null,
    selectedExamProximity: 2,
    activeTheme: "dark",
    lastTopicAnalysis: null,
  };

  const DAY_SHORT = {
    Monday: "Mon",
    Tuesday: "Tue",
    Wednesday: "Wed",
    Thursday: "Thu",
    Friday: "Fri",
    Saturday: "Sat",
    Sunday: "Sun",
  };

  const MODEL_ORDER = ["xgboost", "random_forest", "logistic_regression"];
  const MODEL_COLOR = {
    xgboost: "#5B8CFF",
    random_forest: "#28C7D9",
    logistic_regression: "#FFB44A",
  };

  // Vibrant matplotlib-inspired color palette (18 colors)
  const VIBRANT_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
    "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8B88B", "#A9DFBF",
    "#F5B7B1", "#AED6F1", "#D7BDE2", "#FAD7A0", "#82E0AA",
    "#EC7063", "#52BE80", "#5DADE2"
  ];

  // Extended palette for up to 12 items with vibrant colors
  const EXTENDED_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#F7DC6F",
    "#BB8FCE", "#FF8C42", "#A9DFBF", "#F8B88B", "#E74C3C",
    "#9B59B6", "#3498DB"
  ];

  const THEME = {
    text: "#EAF2FF",
    muted: "#9DB2D8",
    grid: "rgba(157, 178, 216, 0.18)",
  };

  const LIGHT_THEME = {
    text: "#0D1A2E",
    muted: "#4B668F",
    grid: "rgba(66, 95, 138, 0.2)",
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    const node = byId(id);
    if (node) {
      node.textContent = value;
    }
  }

  function showToast(message, type = "info") {
    const toast = byId("toast");
    if (!toast) {
      return;
    }
    toast.classList.remove("hidden");
    toast.textContent = message;

    if (type === "error") {
      toast.style.borderColor = "rgba(255, 110, 116, 0.72)";
    } else {
      toast.style.borderColor = "rgba(142, 174, 232, 0.42)";
    }

    window.clearTimeout(showToast._timer);
    showToast._timer = window.setTimeout(() => {
      toast.classList.add("hidden");
    }, 3800);
  }

  function toDisplayName(modelName) {
    if (!modelName) {
      return "Unknown";
    }
    return modelName
      .split("_")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }

  function formatRate(value, digits = 1) {
    return `${(Number(value || 0) * 100).toFixed(digits)}%`;
  }

  function formatDateTime(iso) {
    if (!iso) {
      return "Updated now";
    }
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) {
      return "Updated now";
    }
    return `Updated ${d.toLocaleString()}`;
  }

  function setTheme(themeName, persist = true) {
    const isLight = themeName === "light";
    document.body.setAttribute("data-theme", isLight ? "light" : "dark");
    state.activeTheme = isLight ? "light" : "dark";

    const btn = byId("themeToggleButton");
    if (btn) {
      btn.textContent = isLight ? "Dark Mode" : "Light Mode";
    }

    if (persist) {
      try {
        window.localStorage.setItem("workshop_theme", state.activeTheme);
      } catch (error) {
        console.warn("Theme persistence unavailable", error);
      }
    }
  }

  function setupThemeToggle() {
    let initialTheme = "dark";
    try {
      const saved = window.localStorage.getItem("workshop_theme");
      if (saved === "light" || saved === "dark") {
        initialTheme = saved;
      }
    } catch (error) {
      console.warn("Theme read unavailable", error);
    }

    setTheme(initialTheme, false);
    const themeToggle = byId("themeToggleButton");
    themeToggle?.addEventListener("click", () => {
      setTheme(state.activeTheme === "light" ? "dark" : "light");
      chartDefaults();
      if (state.chartsPayload) {
        renderEdaCharts(state.chartsPayload);
      }
      if (state.modelDetails) {
        renderModelPage(state.modelDetails);
      }
      if (state.lastTopicAnalysis) {
        renderFilteredTopicInsights(state.lastTopicAnalysis);
      }
    });
  }

  async function apiRequest(path, options = {}) {
    const response = await fetch(`/api${path}`, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `Request failed (${response.status})`);
    }
    return payload;
  }

  function hideSplash(delay = 300) {
    const splash = byId("splash");
    if (!splash) return;
    
    window.setTimeout(() => {
      splash.style.opacity = "0";
      splash.style.visibility = "hidden";
      splash.style.transition = "opacity 0.6s ease-out";
      window.setTimeout(() => {
        if (splash.parentNode) {
          splash.remove();
        }
      }, 620);
    }, delay);
  }

  function updateSplashProgress(text, percentage) {
    const splashText = byId("splashText");
    const splashBar = byId("splashBar");
    if (splashText) splashText.textContent = text;
    if (splashBar) splashBar.style.width = `${Math.min(percentage, 95)}%`;
  }

  function initSplash() {
    const splash = byId("splash");
    const bar = byId("splashBar");
    if (!splash || !bar) {
      return;
    }

    window.requestAnimationFrame(() => {
      bar.style.transition = "width 0.8s cubic-bezier(0.22, 0.61, 0.36, 1)";
      bar.style.width = "30%";
    });
  }

  function setupTabs() {
    const tabButtons = document.querySelectorAll(".tab-button");
    const panels = document.querySelectorAll(".tab-panel");

    function activateTab(tabName) {
      tabButtons.forEach((btn) => {
        btn.classList.toggle("is-active", btn.dataset.tab === tabName);
      });
      panels.forEach((panel) => {
        panel.classList.toggle("is-active", panel.id === tabName);
      });

      if (tabName === "eda") {
        window.setTimeout(() => window.dispatchEvent(new Event("resize")), 120);
      }
    }

    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        activateTab(btn.dataset.tab);
        byId("mobileTabs")?.classList.add("hidden");
      });
    });

    const mobileMenuButton = byId("mobileMenuButton");
    mobileMenuButton?.addEventListener("click", () => {
      byId("mobileTabs")?.classList.toggle("hidden");
    });

    activateTab("predict");
  }

  function setupFormControls() {
    const durationInput = byId("durationInput");
    const registrationsInput = byId("registrationsInput");
    const eventDateInput = byId("eventDateInput");

    if (durationInput) {
      durationInput.addEventListener("input", (event) => {
        setText("durationValue", String(event.target.value));
      });
    }

    if (registrationsInput) {
      registrationsInput.addEventListener("input", (event) => {
        setText("registrationsValue", String(event.target.value));
      });
    }

    if (eventDateInput && !eventDateInput.value) {
      eventDateInput.value = new Date().toISOString().slice(0, 10);
    }

    const examButtons = document.querySelectorAll(".exam-button");
    examButtons.forEach((button) => {
      button.addEventListener("click", () => {
        examButtons.forEach((btn) => btn.classList.remove("is-active"));
        button.classList.add("is-active");
        state.selectedExamProximity = Number(button.dataset.exam || 2);
        const hiddenInput = byId("examProximityInput");
        if (hiddenInput) {
          hiddenInput.value = String(state.selectedExamProximity);
        }
      });
    });
  }

  function fillSelect(selectId, options, fallback = []) {
    const select = byId(selectId);
    if (!select) {
      return;
    }

    const values = Array.isArray(options) && options.length ? options : fallback;
    const previous = select.value;
    select.innerHTML = "";

    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = String(value);
      select.appendChild(option);
    });

    if (previous && values.includes(previous)) {
      select.value = previous;
    }
  }

  function setHealthBadge(isHealthy, message = "") {
    const badge = byId("apiHealthBadge");
    if (!badge) {
      return;
    }
    badge.classList.remove("ok", "bad");
    if (isHealthy) {
      badge.classList.add("ok");
      badge.textContent = "API Connected";
    } else {
      badge.classList.add("bad");
      badge.textContent = message || "API Offline";
    }
  }

  function updatePredictSummary(overview) {
    setText("predictTotalEvents", String(overview.total_events ?? 0));
    setText("predictTotalStudents", String(overview.total_students ?? 0));
    setText("predictRegistrations", Number(overview.registrations || 0).toLocaleString());
    setText("predictAvgAttendance", formatRate(overview.avg_attendance || 0));

    const model = overview.model || {};
    const metrics = model.metrics || {};

    setText("snapshotModel", toDisplayName(model.winner || ""));
    setText("snapshotAccuracy", Number(metrics.accuracy || 0).toFixed(3));
    setText("snapshotF1", Number(metrics.f1_score || 0).toFixed(3));
    setText("snapshotAuc", Number(metrics.auc_roc || 0).toFixed(3));
    setText("snapshotThreshold", Number(model.threshold || metrics.threshold || 0.5).toFixed(2));

    setText("updatedAt", formatDateTime(overview.updated_at));
  }

  function chartDefaults() {
    const palette = state.activeTheme === "light" ? LIGHT_THEME : THEME;
    Chart.defaults.color = palette.muted;
    Chart.defaults.borderColor = palette.grid;
    Chart.defaults.font.family = "Manrope";
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
  }

  function destroyChart(name) {
    if (state.charts[name]) {
      state.charts[name].destroy();
      state.charts[name] = null;
    }
  }

  function renderChart(name, canvasId, config) {
    const canvas = byId(canvasId);
    if (!canvas) {
      return;
    }
    destroyChart(name);
    state.charts[name] = new Chart(canvas, config);
  }

  function baseCartesianOptions(extra = {}) {
    const palette = state.activeTheme === "light" ? LIGHT_THEME : THEME;
    const tooltipBackground = state.activeTheme === "light" ? "rgba(244, 249, 255, 0.96)" : "rgba(9, 16, 30, 0.95)";
    const tooltipBorder = state.activeTheme === "light" ? "rgba(99, 131, 177, 0.35)" : "rgba(142, 174, 232, 0.25)";
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: palette.muted } },
        tooltip: {
          backgroundColor: tooltipBackground,
          borderColor: tooltipBorder,
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          ticks: { color: palette.muted },
          grid: { color: state.activeTheme === "light" ? "rgba(66, 95, 138, 0.08)" : "rgba(157, 178, 216, 0.08)" },
        },
        y: {
          ticks: { color: palette.muted },
          grid: { color: state.activeTheme === "light" ? "rgba(66, 95, 138, 0.11)" : "rgba(157, 178, 216, 0.11)" },
        },
      },
      ...extra,
    };
  }

  function renderHeatmap(heatmapPayload) {
    const heatmapContainer = byId("chartHeatmap");
    if (!heatmapContainer) {
      return;
    }

    const days = heatmapPayload.days || [];
    const slots = heatmapPayload.slots || [];
    const matrix = heatmapPayload.matrix || [];

    if (!days.length || !slots.length) {
      heatmapContainer.innerHTML = '<p class="panel-note">No heatmap data available.</p>';
      return;
    }

    heatmapContainer.innerHTML = ""; // Clear previous content

    const heatmapGrid = document.createElement("div");
    heatmapGrid.className = "heatmap-grid";
    heatmapContainer.appendChild(heatmapGrid);

    heatmapGrid.style.setProperty("--slot-count", String(slots.length));

    const header = document.createElement("div");
    header.className = "heatmap-row";
    const headLabel = document.createElement("div");
    headLabel.className = "heatmap-label";
    headLabel.textContent = "Day / Time";
    header.appendChild(headLabel);

    slots.forEach((slot) => {
      const h = document.createElement("div");
      h.className = "heatmap-head";
      h.textContent = slot;
      header.appendChild(h);
    });

    heatmapGrid.appendChild(header);

    days.forEach((day, dayIndex) => {
      const row = document.createElement("div");
      row.className = "heatmap-row";

      const dayLabel = document.createElement("div");
      dayLabel.className = "heatmap-label";
      dayLabel.textContent = day;
      row.appendChild(dayLabel);

      slots.forEach((slot, slotIndex) => {
        const value = Number((matrix[dayIndex] || [])[slotIndex] || 0);
        const percent = value * 100;

        const cell = document.createElement("div");
        cell.className = "heatmap-cell";
        cell.textContent = `${percent.toFixed(1)}%`;

        const strength = Math.max(0.12, Math.min(0.9, value));
        cell.style.backgroundColor = `rgba(40, 199, 217, ${strength.toFixed(3)})`;
        cell.style.color = percent > 50 ? "#08222b" : "#eaf6ff";
        cell.title = `${day} • ${slot} -> ${percent.toFixed(1)}%`;

        row.appendChild(cell);
      });

      heatmapGrid.appendChild(row);
    });
  }

  function renderEdaCharts(payload) {
    // 1. Topic Rankings - Horizontal Bar with Vibrant Colors
    const topicLabels = payload.topic?.labels || [];
    const topicRates = (payload.topic?.rates || []).map((v) => Number(v) * 100);
    renderChart("topic", "chartTopic", {
      type: "bar",
      data: {
        labels: topicLabels,
        datasets: [{
          label: "Attendance %",
          data: topicRates,
          backgroundColor: EXTENDED_PALETTE.slice(0, Math.max(5, Math.min(12, topicLabels.length))),
          borderRadius: 8,
          borderWidth: 2,
          borderColor: "rgba(255,255,255,0.4)",
        }],
      },
      options: baseCartesianOptions({
        indexAxis: "y",
        plugins: { legend: { display: false } },
        onClick: (_event, elements, chart) => {
          if (!elements.length) return;
          const index = elements[0].index;
          const topic = chart.data.labels?.[index];
          if (!topic) return;
          const topicFilter = byId("edaTopicFilter");
          if (topicFilter) topicFilter.value = String(topic);
          applyEdaFilters();
          showToast(`Topic filter applied: ${topic}`);
        },
      }),
    });

    // 2. Day of Week - Bar Chart with Weekend Colors
    const dayLabels = (payload.day?.labels || []).map((d) => DAY_SHORT[d] || d);
    const dayRates = (payload.day?.rates || []).map((v) => Number(v) * 100);
    renderChart("day", "chartDay", {
      type: "bar",
      data: {
        labels: dayLabels,
        datasets: [{
          label: "Attendance %",
          data: dayRates,
          backgroundColor: dayLabels.map((d) => (d === "Sat" || d === "Sun" ? "#FF6B6B" : "#4ECDC4")),
          borderRadius: 8,
          borderWidth: 2,
          borderColor: "rgba(255,255,255,0.3)",
        }],
      },
      options: baseCartesianOptions({ plugins: { legend: { display: false } } }),
    });

    // 3. Monthly Trends - Multi-line + Bar
    renderChart("month", "chartMonth", {
      data: {
        labels: payload.monthly?.labels || [],
        datasets: [
          {
            type: "line",
            label: "Registrations",
            data: payload.monthly?.registrations || [],
            borderColor: "#FF6B6B",
            backgroundColor: "rgba(255, 107, 107, 0.15)",
            tension: 0.4,
            pointRadius: 6,
            pointBorderColor: "#FF6B6B",
            pointBackgroundColor: "#FF6B6B",
            pointBorderWidth: 2,
            yAxisID: "y",
          },
          {
            type: "line",
            label: "Attended",
            data: payload.monthly?.attended || [],
            borderColor: "#4ECDC4",
            backgroundColor: "rgba(78, 205, 196, 0.15)",
            tension: 0.4,
            pointRadius: 6,
            pointBorderColor: "#4ECDC4",
            pointBackgroundColor: "#4ECDC4",
            pointBorderWidth: 2,
            yAxisID: "y",
          },
          {
            type: "bar",
            label: "Rate %",
            data: (payload.monthly?.rates || []).map((v) => Number(v) * 100),
            backgroundColor: "#45B7D1",
            borderRadius: 6,
            yAxisID: "y1",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.muted, padding: 15 } },
        },
        scales: {
          x: { ticks: { color: THEME.muted }, grid: { color: "rgba(157, 178, 216, 0.08)" } },
          y: { ticks: { color: THEME.muted }, grid: { color: "rgba(157, 178, 216, 0.11)" } },
          y1: {
            position: "right",
            min: 0,
            max: 100,
            grid: { drawOnChartArea: false },
            ticks: { color: THEME.muted },
          },
        },
      },
    });

    // 4. Mode Radar Chart
    renderChart("modeRadar", "chartModeRadar", {
      type: "radar",
      data: {
        labels: payload.mode?.labels || [],
        datasets: [{
          label: "Attendance Rate",
          data: (payload.mode?.rates || []).map((v) => Number(v) * 100),
          borderColor: "#4ECDC4",
          backgroundColor: "rgba(78, 205, 196, 0.25)",
          pointBackgroundColor: "#4ECDC4",
          pointBorderColor: "#fff",
          pointBorderWidth: 2,
          pointRadius: 6,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.muted, padding: 15 } },
        },
        scales: {
          r: {
            grid: { color: "rgba(157, 178, 216, 0.2)" },
            ticks: { color: THEME.muted },
          },
        },
      },
    });

    // 5. Department Radar Chart
    renderChart("deptRadar", "chartDeptRadar", {
      type: "radar",
      data: {
        labels: payload.department?.labels || [],
        datasets: [{
          label: "Attendance Rate",
          data: (payload.department?.rates || []).map((v) => Number(v) * 100),
          borderColor: "#FF6B6B",
          backgroundColor: "rgba(255, 107, 107, 0.25)",
          pointBackgroundColor: "#FF6B6B",
          pointBorderColor: "#fff",
          pointBorderWidth: 2,
          pointRadius: 6,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.muted, padding: 15 } },
        },
        scales: {
          r: {
            grid: { color: "rgba(157, 178, 216, 0.2)" },
            ticks: { color: THEME.muted },
          },
        },
      },
    });

    // 6. Speaker Impact - Vibrant Bars
    renderChart("speaker", "chartSpeaker", {
      type: "bar",
      data: {
        labels: payload.speaker?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.speaker?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: EXTENDED_PALETTE.slice(0, Math.max(3, Math.min(12, (payload.speaker?.labels || []).length))),
          borderRadius: 8,
          borderWidth: 2,
          borderColor: "rgba(255,255,255,0.3)",
        }],
      },
      options: baseCartesianOptions({
        indexAxis: "y",
        plugins: { legend: { display: false } },
      }),
    });

    // 7. Mode Pie Chart
    renderChart("modePie", "chartModePie", {
      type: "doughnut",
      data: {
        labels: payload.mode?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.mode?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#F7DC6F"],
          borderColor: "rgba(255,255,255,0.4)",
          borderWidth: 3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.muted, padding: 15, font: { size: 12, weight: 600 } } },
          tooltip: { backgroundColor: "rgba(0,0,0,0.9)", titleColor: "#fff", bodyColor: "#fff", padding: 12 },
        },
      },
    });

    // 8. Club Activity - Vibrant Bars
    renderChart("club", "chartClub", {
      type: "bar",
      data: {
        labels: payload.club?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.club?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: EXTENDED_PALETTE.slice(0, Math.max(3, Math.min(12, (payload.club?.labels || []).length))),
          borderRadius: 8,
          borderWidth: 2,
          borderColor: "rgba(255,255,255,0.3)",
        }],
      },
      options: baseCartesianOptions({ plugins: { legend: { display: false } } }),
    });

    // 9. Exam Proximity - Gradient Colors
    const examLabels = (payload.exam?.labels || []).map((v) => {
      if (Number(v) === 1) return "Near";
      if (Number(v) === 2) return "Moderate";
      return "Far";
    });
    renderChart("exam", "chartExam", {
      type: "bar",
      data: {
        labels: examLabels,
        datasets: [{
          label: "Attendance %",
          data: (payload.exam?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: ["#FF6B6B", "#F7DC6F", "#1FD199"],
          borderRadius: 8,
          borderWidth: 2,
          borderColor: "rgba(255,255,255,0.3)",
        }],
      },
      options: baseCartesianOptions({ plugins: { legend: { display: false } } }),
    });

    // 10. Department Stacked Bar Chart
    renderChart("deptStacked", "chartDeptStacked", {
      type: "bar",
      data: {
        labels: payload.department?.labels || [],
        datasets: [
          {
            label: "Attended",
            data: (payload.department?.rates || []).map((v) => Number(v) * 100),
            backgroundColor: "#4ECDC4",
            borderRadius: 6,
          },
          {
            label: "Not Attended",
            data: (payload.department?.rates || []).map((v) => (1 - Number(v)) * 100),
            backgroundColor: "rgba(157, 178, 216, 0.3)",
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        plugins: {
          legend: { labels: { color: THEME.muted, padding: 15 } },
        },
        scales: {
          x: {
            stacked: true,
            ticks: { color: THEME.muted },
            grid: { color: "rgba(157, 178, 216, 0.08)" },
          },
          y: {
            stacked: true,
            ticks: { color: THEME.muted },
            grid: { color: "rgba(157, 178, 216, 0.11)" },
          },
        },
      },
    });

    // 11. Semester Distribution - Pie Chart
    renderChart("semesterPie", "chartSemesterPie", {
      type: "doughnut",
      data: {
        labels: payload.semester?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.semester?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: ["#45B7D1", "#BB8FCE", "#F8B88B", "#85C1E9"],
          borderColor: "rgba(255,255,255,0.4)",
          borderWidth: 3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.muted, padding: 15, font: { size: 12, weight: 600 } } },
          tooltip: { backgroundColor: "rgba(0,0,0,0.9)", titleColor: "#fff", bodyColor: "#fff", padding: 12 },
        },
      },
    });

    renderHeatmap(payload.heatmap || {});
  }

  function renderFilteredTopicInsights(payload) {
    state.lastTopicAnalysis = payload;
    const summary = payload.summary || {};
    setText(
      "topicInsightSummary",
      `${payload.selected_topic || "All Topics"}: ${formatRate(summary.rate || 0, 1)} attendance across ${summary.events || 0} events and ${summary.registrations || 0} registrations.`,
    );

    // Department Bar Chart
    renderChart("topicFilterDept", "chartTopicFilterDept", {
      type: "bar",
      data: {
        labels: payload.department?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.department?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: EXTENDED_PALETTE.slice(0, Math.max(3, Math.min(12, (payload.department?.labels || []).length))),
          borderRadius: 8,
          borderWidth: 2,
          borderColor: "rgba(255,255,255,0.4)",
        }],
      },
      options: baseCartesianOptions({
        indexAxis: "y",
        plugins: { legend: { display: false } },
      }),
    });

    // Mode Doughnut Chart with enhanced styling
    renderChart("topicFilterMode", "chartTopicFilterMode", {
      type: "doughnut",
      data: {
        labels: payload.mode?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.mode?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#F7DC6F"],
          borderColor: "rgba(255,255,255,0.5)",
          borderWidth: 3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.muted, padding: 15, font: { size: 11, weight: 600 } } },
          tooltip: { backgroundColor: "rgba(0,0,0,0.9)", titleColor: "#fff", bodyColor: "#fff", padding: 10 },
        },
      },
    });

    // Club Activity Bar Chart
    renderChart("topicFilterClub", "chartTopicFilterClub", {
      type: "bar",
      data: {
        labels: payload.club?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.club?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: EXTENDED_PALETTE.slice(0, Math.max(3, Math.min(12, (payload.club?.labels || []).length))),
          borderRadius: 8,
          borderWidth: 2,
          borderColor: "rgba(255,255,255,0.4)",
        }],
      },
      options: baseCartesianOptions({ plugins: { legend: { display: false } } }),
    });
  }

  function setupEdaFilters(options, charts) {
    fillSelect("edaTopicFilter", ["All Topics", ...(options.topics || [])], ["All Topics"]);
    fillSelect("edaSchoolFilter", ["All Schools", ...(charts.department?.labels || [])], ["All Schools"]);

    const applyButton = byId("applyEdaFilterButton");
    const clearButton = byId("clearEdaFilterButton");

    applyButton?.addEventListener("click", () => {
      applyEdaFilters();
    });

    clearButton?.addEventListener("click", () => {
      const topicFilter = byId("edaTopicFilter");
      const schoolFilter = byId("edaSchoolFilter");
      if (topicFilter) {
        topicFilter.value = "All Topics";
      }
      if (schoolFilter) {
        schoolFilter.value = "All Schools";
      }
      applyEdaFilters();
    });
  }

  async function applyEdaFilters() {
    const topic = byId("edaTopicFilter")?.value || "All Topics";
    const school = byId("edaSchoolFilter")?.value || "All Schools";

    const query = new URLSearchParams();
    if (topic && topic !== "All Topics") {
      query.set("topic", topic);
    }
    if (school && school !== "All Schools") {
      query.set("school", school);
    }

    try {
      const path = query.toString() ? `/topic-analysis?${query.toString()}` : "/topic-analysis";
      const payload = await apiRequest(path);
      renderFilteredTopicInsights(payload);
    } catch (error) {
      showToast(`Filter apply failed: ${error.message}`, "error");
      console.error(error);
    }
  }

  function renderModelPage(modelDetails) {
    const metrics = modelDetails.metrics || {};
    setText("nerdWinnerModel", modelDetails.winner_display || toDisplayName(modelDetails.winner || ""));
    setText("nerdAccuracy", Number(metrics.accuracy || 0).toFixed(3));
    setText("nerdF1", Number(metrics.f1_score || 0).toFixed(3));
    setText("nerdAuc", Number(metrics.auc_roc || 0).toFixed(3));
    setText("nerdThreshold", Number(modelDetails.threshold || metrics.threshold || 0.5).toFixed(2));
    setText("nerdFeatureCount", String(modelDetails.feature_count || 0));

    const comparison = modelDetails.comparison || {};
    const tableBody = byId("modelComparisonBody");
    if (tableBody) {
      tableBody.innerHTML = "";
      MODEL_ORDER.forEach((modelName) => {
        const rowData = comparison[modelName];
        if (!rowData || typeof rowData !== "object") {
          return;
        }
        const row = document.createElement("tr");
        if (modelName === modelDetails.winner) {
          row.className = "winner-row";
        }
        row.innerHTML = `
          <td>${toDisplayName(modelName)}${modelName === modelDetails.winner ? " (Winner)" : ""}</td>
          <td>${Number(rowData.accuracy || 0).toFixed(4)}</td>
          <td>${Number(rowData.f1_score || 0).toFixed(4)}</td>
          <td>${Number(rowData.auc_roc || 0).toFixed(4)}</td>
          <td>${Number(rowData.threshold || 0.5).toFixed(2)}</td>
        `;
        tableBody.appendChild(row);
      });
    }

    const featuresList = byId("topFeaturesList");
    if (featuresList) {
      featuresList.innerHTML = "";
      const features = modelDetails.top_features || [];
      if (!features.length) {
        const li = document.createElement("li");
        li.textContent = "Feature list unavailable in metadata.";
        featuresList.appendChild(li);
      } else {
        features.forEach((feature) => {
          const li = document.createElement("li");
          li.textContent = feature;
          featuresList.appendChild(li);
        });
      }
    }

    const timelineBody = byId("maintenanceTimelineBody");
    if (timelineBody) {
      timelineBody.innerHTML = "";
      (modelDetails.maintenance_timeline || []).forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${item.phase || "-"}</td>
          <td>${item.frequency || "-"}</td>
          <td>${item.trigger || "-"}</td>
          <td>${item.action || "-"}</td>
        `;
        timelineBody.appendChild(row);
      });
    }

    const retrainHistory = modelDetails.retrain_history || [];
    setText("retrainHistory", retrainHistory.length ? retrainHistory.join("\n") : "No retraining attempts yet.");

    const labels = ["Accuracy", "F1 Score", "AUC-ROC"];
    const datasetByMetric = {
      accuracy: MODEL_ORDER.map((name) => Number(comparison[name]?.accuracy || 0) * 100),
      f1: MODEL_ORDER.map((name) => Number(comparison[name]?.f1_score || 0) * 100),
      auc: MODEL_ORDER.map((name) => Number(comparison[name]?.auc_roc || 0) * 100),
    };

    renderChart("modelComparison", "chartModelComparison", {
      type: "bar",
      data: {
        labels,
        datasets: MODEL_ORDER.map((modelName) => ({
          label: toDisplayName(modelName),
          data: [
            Number(comparison[modelName]?.accuracy || 0) * 100,
            Number(comparison[modelName]?.f1_score || 0) * 100,
            Number(comparison[modelName]?.auc_roc || 0) * 100,
          ],
          backgroundColor: MODEL_COLOR[modelName],
          borderRadius: 8,
        })),
      },
      options: baseCartesianOptions({
        scales: {
          x: { ticks: { color: THEME.muted }, grid: { color: "rgba(157,178,216,0.08)" } },
          y: { ticks: { color: THEME.muted }, grid: { color: "rgba(157,178,216,0.11)" }, min: 0, max: 100 },
        },
      }),
    });

    renderChart("modelRadar", "chartModelRadar", {
      type: "radar",
      data: {
        labels: ["Accuracy", "F1 Score", "AUC-ROC", "Calibration", "Stability", "Operational Fit"],
        datasets: [
          {
            label: modelDetails.winner_display || "Winner",
            data: [
              Number(metrics.accuracy || 0) * 100,
              Number(metrics.f1_score || 0) * 100,
              Number(metrics.auc_roc || 0) * 100,
              Number(metrics.auc_roc || 0) * 95,
              Number(metrics.f1_score || 0) * 92,
              Math.max(65, Number(metrics.f1_score || 0) * 100),
            ],
            borderColor: "#28C7D9",
            backgroundColor: "rgba(40, 199, 217, 0.22)",
            pointBackgroundColor: "#28C7D9",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.muted } },
        },
        scales: {
          r: {
            min: 0,
            max: 100,
            ticks: { display: false },
            pointLabels: { color: THEME.muted },
            grid: { color: "rgba(157, 178, 216, 0.12)" },
            angleLines: { color: "rgba(157, 178, 216, 0.14)" },
          },
        },
      },
    });
  }

  async function loadInitialData() {
    try {
      updateSplashProgress("Loading health...", 15);
      const health = await apiRequest("/health");
      setHealthBadge(Boolean(health.ok));
      
      updateSplashProgress("Loading overview...", 30);
      const overview = await apiRequest("/overview");
      state.overview = overview;
      updatePredictSummary(overview);
      
      updateSplashProgress("Loading options...", 45);
      const options = await apiRequest("/options");
      state.options = options;
      
      updateSplashProgress("Loading charts...", 60);
      const charts = await apiRequest("/charts");
      state.chartsPayload = charts;
      
      updateSplashProgress("Loading model...", 75);
      const modelDetails = await apiRequest("/model-details");
      state.modelDetails = modelDetails;

      fillSelect("topicInput", options.topics || []);
      fillSelect("speakerTypeInput", options.speaker_types || []);
      fillSelect("dayInput", options.days || []);
      fillSelect("timeSlotInput", options.time_slots || []);
      fillSelect("modeInput", options.modes || ["Offline", "Online"]);
      fillSelect("promotionLevelInput", options.promotion_levels || ["Low", "Medium", "High"]);
      
      updateSplashProgress("Rendering visualizations...", 85);
      setupEdaFilters(options, charts);
      renderEdaCharts(charts);
      await applyEdaFilters();
      renderModelPage(modelDetails);
      
      updateSplashProgress("Ready!", 100);
      hideSplash(100);
    } catch (error) {
      setHealthBadge(false, "API Error");
      updateSplashProgress(`Error: ${error.message}`, 100);
      showToast(`Initialization failed: ${error.message}`, "error");
      console.error(error);
      hideSplash(2000);
    }
  }

  async function fetchTopicReference(topicName) {
    if (!topicName) {
      return null;
    }

    try {
      const payload = await apiRequest(`/topic-analysis?topic=${encodeURIComponent(topicName)}`);
      return payload;
    } catch (error) {
      console.warn("Failed topic analysis lookup", error);
      return null;
    }
  }

  function setPredictionAdvice(predictedAttendees) {
    const advice = byId("resultAdvice");
    if (!advice) {
      return;
    }

    advice.classList.remove("good", "warn", "bad", "neutral");

    if (predictedAttendees < 20) {
      advice.textContent = "Low predicted turnout. Consider stronger promotion, speaker switch, or schedule adjustment.";
      advice.classList.add("bad");
    } else if (predictedAttendees < 40) {
      advice.textContent = "Moderate turnout expected. Mid-size room and standard support should be enough.";
      advice.classList.add("warn");
    } else {
      advice.textContent = "High turnout expected. Prepare larger venue capacity and additional engagement support.";
      advice.classList.add("good");
    }
  }

  function applyPredictionRisk(rate) {
    const box = byId("predictionResult");
    if (!box) {
      return;
    }

    box.classList.remove("risk-high", "risk-medium", "risk-low");

    if (rate >= 0.7) {
      box.classList.add("risk-high");
    } else if (rate >= 0.4) {
      box.classList.add("risk-medium");
    } else {
      box.classList.add("risk-low");
    }
  }

  function updatePredictionRing(attendanceRate) {
    const ring = byId("predictionRing");
    if (!ring) {
      return;
    }
    const circumference = 2 * Math.PI * 86;
    const offset = circumference * (1 - attendanceRate);
    ring.style.strokeDasharray = String(circumference);
    ring.style.strokeDashoffset = String(offset);
  }

  function setupPredictionSubmit() {
    const form = byId("predictionForm");
    if (!form) {
      return;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const button = byId("predictButton");
      const buttonText = byId("predictButtonText");
      const loader = byId("predictLoader");

      button?.setAttribute("disabled", "true");
      loader?.classList.remove("hidden");
      if (buttonText) {
        buttonText.textContent = "Predicting...";
      }

      const payload = {
        topic: byId("topicInput")?.value || "",
        speaker_type: byId("speakerTypeInput")?.value || "",
        day_of_week: byId("dayInput")?.value || "",
        time_slot: byId("timeSlotInput")?.value || "",
        mode: byId("modeInput")?.value || "Offline",
        duration_minutes: Number(byId("durationInput")?.value || 90),
        exam_proximity: Number(byId("examProximityInput")?.value || state.selectedExamProximity || 2),
        promotion_level: byId("promotionLevelInput")?.value || "Medium",
        num_registrations: Number(byId("registrationsInput")?.value || 50),
        event_date: byId("eventDateInput")?.value || new Date().toISOString().slice(0, 10),
      };

      try {
        const result = await apiRequest("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        byId("predictionEmpty")?.classList.add("hidden");
        byId("predictionResult")?.classList.remove("hidden");

        const rate = Number(result.attendance_rate || 0);
        const rateText = formatRate(rate, 1);
        setText("ringPercent", rateText);
        setText("resultRate", rateText);
        setText("resultRegistered", String(result.total_registered || 0));
        setText("resultAttendees", String(result.predicted_attendees || 0));

        const confidenceLabel = result.confidence || "Medium";
        const probability = Number(result.avg_probability || 0);
        setText("resultConfidence", confidenceLabel);
        setText("resultProbability", `Average model probability: ${formatRate(probability, 1)} (${confidenceLabel})`);

        applyPredictionRisk(rate);
        updatePredictionRing(rate);
        setPredictionAdvice(Number(result.predicted_attendees || 0));

        const topicReference = await fetchTopicReference(payload.topic);
        if (topicReference?.summary) {
          const summary = topicReference.summary;
          setText(
            "historicalReference",
            `${payload.topic} historical rate: ${formatRate(summary.rate || 0, 1)} across ${summary.events || 0} events.`,
          );
        } else {
          setText("historicalReference", "Historical topic reference currently unavailable.");
        }
      } catch (error) {
        showToast(`Prediction failed: ${error.message}`, "error");
        console.error(error);
      } finally {
        button?.removeAttribute("disabled");
        loader?.classList.add("hidden");
        if (buttonText) {
          buttonText.textContent = "Predict Attendance";
        }
      }
    });
  }

  function init() {
    initSplash();
    setupThemeToggle();
    setupTabs();
    setupFormControls();
    setupPredictionSubmit();
    chartDefaults();
    loadInitialData();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
