(() => {
  const state = {
    charts: {},
    options: null,
    chartsPayload: null,
    overview: null,
    modelDetails: null,
    selectedExamProximity: 2,
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

  const THEME = {
    text: "#EAF2FF",
    muted: "#9DB2D8",
    grid: "rgba(157, 178, 216, 0.18)",
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

  async function apiRequest(path, options = {}) {
    const response = await fetch(`/api${path}`, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `Request failed (${response.status})`);
    }
    return payload;
  }

  function initSplash() {
    const splash = byId("splash");
    const bar = byId("splashBar");
    if (!splash || !bar) {
      return;
    }

    window.requestAnimationFrame(() => {
      bar.style.transition = "width 1.5s cubic-bezier(0.22, 0.61, 0.36, 1)";
      bar.style.width = "100%";
    });

    window.setTimeout(() => {
      splash.style.opacity = "0";
      splash.style.visibility = "hidden";
      window.setTimeout(() => {
        splash.remove();
      }, 620);
    }, 1700);
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
    Chart.defaults.color = THEME.muted;
    Chart.defaults.borderColor = THEME.grid;
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
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: THEME.muted } },
        tooltip: {
          backgroundColor: "rgba(9, 16, 30, 0.95)",
          borderColor: "rgba(142, 174, 232, 0.25)",
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          ticks: { color: THEME.muted },
          grid: { color: "rgba(157, 178, 216, 0.08)" },
        },
        y: {
          ticks: { color: THEME.muted },
          grid: { color: "rgba(157, 178, 216, 0.11)" },
        },
      },
      ...extra,
    };
  }

  function renderHeatmap(heatmapPayload) {
    const heatmapGrid = byId("heatmapGrid");
    if (!heatmapGrid) {
      return;
    }

    const days = heatmapPayload.days || [];
    const slots = heatmapPayload.slots || [];
    const matrix = heatmapPayload.matrix || [];

    if (!days.length || !slots.length) {
      heatmapGrid.innerHTML = "<p class=\"panel-note\">No heatmap data available.</p>";
      return;
    }

    heatmapGrid.innerHTML = "";
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
    const topicLabels = payload.topic?.labels || [];
    const topicRates = (payload.topic?.rates || []).map((v) => Number(v) * 100);
    renderChart("topic", "chartTopic", {
      type: "bar",
      data: {
        labels: topicLabels,
        datasets: [{
          label: "Attendance %",
          data: topicRates,
          backgroundColor: "#5B8CFF",
          borderRadius: 8,
        }],
      },
      options: baseCartesianOptions({
        indexAxis: "y",
        plugins: { legend: { display: false } },
      }),
    });

    const dayLabels = (payload.day?.labels || []).map((d) => DAY_SHORT[d] || d);
    const dayRates = (payload.day?.rates || []).map((v) => Number(v) * 100);
    renderChart("day", "chartDay", {
      type: "bar",
      data: {
        labels: dayLabels,
        datasets: [{
          label: "Attendance %",
          data: dayRates,
          backgroundColor: dayLabels.map((d) => (d === "Sat" || d === "Sun" ? "#FF7F66" : "#28C7D9")),
          borderRadius: 8,
        }],
      },
      options: baseCartesianOptions({ plugins: { legend: { display: false } } }),
    });

    renderChart("month", "chartMonth", {
      data: {
        labels: payload.monthly?.labels || [],
        datasets: [
          {
            type: "line",
            label: "Registrations",
            data: payload.monthly?.registrations || [],
            borderColor: "#5B8CFF",
            backgroundColor: "rgba(91, 140, 255, 0.14)",
            tension: 0.3,
            yAxisID: "y",
          },
          {
            type: "line",
            label: "Attended",
            data: payload.monthly?.attended || [],
            borderColor: "#28C7D9",
            backgroundColor: "rgba(40, 199, 217, 0.12)",
            tension: 0.3,
            yAxisID: "y",
          },
          {
            type: "bar",
            label: "Rate %",
            data: (payload.monthly?.rates || []).map((v) => Number(v) * 100),
            backgroundColor: "rgba(255, 180, 74, 0.4)",
            borderRadius: 6,
            yAxisID: "y1",
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

    renderChart("speaker", "chartSpeaker", {
      type: "bar",
      data: {
        labels: payload.speaker?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.speaker?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: "#5B8CFF",
          borderRadius: 8,
        }],
      },
      options: baseCartesianOptions({
        indexAxis: "y",
        plugins: { legend: { display: false } },
      }),
    });

    renderChart("mode", "chartMode", {
      type: "bar",
      data: {
        labels: payload.mode?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.mode?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: ["#5B8CFF", "#28C7D9"],
          borderRadius: 8,
        }],
      },
      options: baseCartesianOptions({ plugins: { legend: { display: false } } }),
    });

    renderChart("club", "chartClub", {
      type: "bar",
      data: {
        labels: payload.club?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.club?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: ["#FF7F66", "#FFB44A", "#1FD199"],
          borderRadius: 8,
        }],
      },
      options: baseCartesianOptions({ plugins: { legend: { display: false } } }),
    });

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
          backgroundColor: ["#FF7F66", "#FFB44A", "#1FD199"],
          borderRadius: 8,
        }],
      },
      options: baseCartesianOptions({ plugins: { legend: { display: false } } }),
    });

    renderChart("dept", "chartDept", {
      type: "bar",
      data: {
        labels: payload.department?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.department?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: "#28C7D9",
          borderRadius: 8,
        }],
      },
      options: baseCartesianOptions({
        indexAxis: "y",
        plugins: { legend: { display: false } },
      }),
    });

    renderChart("semester", "chartSemester", {
      type: "bar",
      data: {
        labels: payload.semester?.labels || [],
        datasets: [{
          label: "Attendance %",
          data: (payload.semester?.rates || []).map((v) => Number(v) * 100),
          backgroundColor: "#5B8CFF",
          borderRadius: 8,
        }],
      },
      options: baseCartesianOptions({ plugins: { legend: { display: false } } }),
    });

    renderHeatmap(payload.heatmap || {});
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
      const [health, overview, options, charts, modelDetails] = await Promise.all([
        apiRequest("/health"),
        apiRequest("/overview"),
        apiRequest("/options"),
        apiRequest("/charts"),
        apiRequest("/model-details"),
      ]);

      setHealthBadge(Boolean(health.ok));
      state.overview = overview;
      state.options = options;
      state.chartsPayload = charts;
      state.modelDetails = modelDetails;

      updatePredictSummary(overview);

      fillSelect("topicInput", options.topics || []);
      fillSelect("speakerTypeInput", options.speaker_types || []);
      fillSelect("dayInput", options.days || []);
      fillSelect("timeSlotInput", options.time_slots || []);
      fillSelect("modeInput", options.modes || ["Offline", "Online"]);
      fillSelect("promotionLevelInput", options.promotion_levels || ["Low", "Medium", "High"]);

      renderEdaCharts(charts);
      renderModelPage(modelDetails);
    } catch (error) {
      setHealthBadge(false, "API Error");
      showToast(`Initialization failed: ${error.message}`, "error");
      console.error(error);
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
    setupTabs();
    setupFormControls();
    setupPredictionSubmit();
    chartDefaults();
    loadInitialData();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
