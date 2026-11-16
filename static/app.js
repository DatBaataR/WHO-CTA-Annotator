/**
 * WHO Epidemiological Column Type Annotator & Symptom Extraction Suite — Frontend Controller
 */

let APP_STATE = {
  presets: {},
  allClasses: [],
  session: { loaded: false },
  symptomPreview: {
    colIndex: null,
    targetCol: "",
    items: [], // { name, true_count, selected, isCustom }
  },
};

function showLoading(title = "Processing...", subtitle = "Please wait") {
  document.getElementById("loading-title").textContent = title;
  document.getElementById("loading-subtitle").textContent = subtitle;
  document.getElementById("loading-overlay").classList.remove("hidden");
}

function hideLoading() {
  document.getElementById("loading-overlay").classList.add("hidden");
}

let toastTimer = null;
function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 3200);
}

async function apiPost(url, payload = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

function getSelectedRoutingParams() {
  return {
    mode: document.getElementById("mode-select").value,
    conf_threshold: parseFloat(document.getElementById("input-thresh").value),
    ensemble_tolerance: parseInt(document.getElementById("input-tol").value, 10),
    use_weak_classes: document.getElementById("chk-weak-classes").checked,
    use_soft_penalty: document.getElementById("chk-soft-penalty").checked,
    auto_apply_predictions: document.getElementById("chk-auto-accept").checked,
    api_key: document.getElementById("input-api-key").value.trim() || null,
  };
}

function syncPresetControls() {
  const mode = document.getElementById("mode-select").value;
  const preset = APP_STATE.presets[mode];
  if (!preset) return;
  document.getElementById("input-thresh").value = preset.conf_threshold.toFixed(2);
  document.getElementById("input-tol").value = preset.ensemble_tolerance;
  document.getElementById("chk-weak-classes").checked = !!preset.use_weak_classes;
  document.getElementById("chk-soft-penalty").checked = !!preset.use_soft_penalty;
  const descEl = document.getElementById("preset-description");
  if (descEl) {
    descEl.textContent = preset.description || "";
  }
}

async function initStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    APP_STATE.presets = data.presets || {};
    APP_STATE.allClasses = data.all_classes || [];

    // Populate hardware status badges
    const hwContainer = document.getElementById("hw-badges");
    hwContainer.innerHTML = "";
    const recaPill = document.createElement("span");
    if (data.reca_available) {
      recaPill.className = "hw-pill ok";
      recaPill.textContent = `RECA Ensemble: Ready (${data.reca_device.toUpperCase()})`;
    } else {
      recaPill.className = "hw-pill sub";
      recaPill.textContent = "RECA Checkpoints Not Installed (Using RF Substitute)";
      // Default to rf_pareto if RECA weights aren't downloaded yet
      document.getElementById("mode-select").value = "rf_pareto";
    }
    hwContainer.appendChild(recaPill);

    const rfPill = document.createElement("span");
    rfPill.className = "hw-pill ok";
    rfPill.textContent = "RF Substitute: Ready (CPU)";
    hwContainer.appendChild(rfPill);

    // Populate datalist of 73 WHO classes
    const dl = document.getElementById("who-semantic-classes");
    dl.innerHTML = "";
    APP_STATE.allClasses.forEach((clsName) => {
      const opt = document.createElement("option");
      opt.value = clsName;
      dl.appendChild(opt);
    });

    // Populate sample files dropdown
    const sampleSelect = document.getElementById("sample-select");
    (data.sample_files || []).forEach((sName) => {
      const opt = document.createElement("option");
      opt.value = sName;
      opt.textContent = `Sample: ${sName}`;
      sampleSelect.appendChild(opt);
    });

    syncPresetControls();

    if (data.session && data.session.loaded) {
      renderSession(data.session);
    }
  } catch (err) {
    console.error("Failed to initialize status:", err);
  }
}

function renderSession(session, promptSymptomsIfFound = false) {
  APP_STATE.session = session;

  const btnUndo = document.getElementById("btn-undo");
  const btnRedo = document.getElementById("btn-redo");
  const btnExport = document.getElementById("btn-export");
  const btnAnnotate = document.getElementById("btn-run-annotate");

  btnUndo.disabled = !(session.undo_count > 0);
  if (session.last_undo_label) {
    btnUndo.title = `Revert: ${session.last_undo_label} (Ctrl + Z)`;
  }
  btnRedo.disabled = !(session.redo_count > 0);
  btnExport.disabled = !session.loaded;
  btnAnnotate.disabled = !session.loaded;

  if (!session.loaded) {
    document.getElementById("empty-state").classList.remove("hidden");
    document.getElementById("grid-wrapper").classList.add("hidden");
    document.getElementById("table-toolbar").classList.add("hidden");
    return;
  }

  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("grid-wrapper").classList.remove("hidden");
  document.getElementById("table-toolbar").classList.remove("hidden");

  // Update toolbar stats
  document.getElementById("stat-filename").textContent = `File: ${session.filename}`;
  document.getElementById("stat-dims").textContent = `${session.total_rows} Rows × ${session.total_cols} Columns`;

  const info = session.annotation_info || {};
  const routingEl = document.getElementById("stat-routing");
  if (info.annotated) {
    const engineName = info.backbone_used === "reca" ? "RECA Ensemble" : "Pruned RF";
    const modeTag = info.hybrid_enabled
      ? `${engineName} + Hybrid (θ=${Number(info.conf_threshold).toFixed(2)}, τ=${info.ensemble_tolerance})`
      : `Pure ${engineName}`;
    routingEl.textContent = modeTag;
    routingEl.classList.remove("hidden");
  } else {
    routingEl.classList.add("hidden");
  }

  // Fallback notice banner
  const fbBanner = document.getElementById("fallback-banner");
  if (info.fallback_notice) {
    document.getElementById("fallback-text").textContent = info.fallback_notice;
    fbBanner.classList.remove("hidden");
  } else {
    fbBanner.classList.add("hidden");
  }

  // Symptoms detection prompt banner
  const symCandidates = session.symptoms_candidates || [];
  const symBanner = document.getElementById("symptoms-prompt-banner");
  if (symCandidates.length > 0) {
    const firstSym = symCandidates[0];
    document.getElementById("symptoms-prompt-text").innerHTML =
      `Column <strong>"${escapeHtml(firstSym.header)}"</strong> (#${firstSym.col_index + 1}) is annotated as <code>Symptoms</code>. ` +
      `Click <strong>Preview &amp; Extract Symptom Columns</strong> to inspect the generated <code>symptom: &lt;name&gt;</code> boolean columns, or add/remove columns effortlessly.`;
    symBanner.classList.remove("hidden");
    symBanner.dataset.colIndex = firstSym.col_index;
  } else {
    symBanner.classList.add("hidden");
  }

  renderDataGrid(session);

  if (promptSymptomsIfFound && symCandidates.length > 0) {
    openSymptomExtractionModal(symCandidates[0].col_index);
  }
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderDataGrid(session) {
  const thead = document.getElementById("grid-thead");
  const tbody = document.getElementById("grid-tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const colsMeta = session.columns_meta || [];

  // Row 1: Model Prediction, Confidence, Routing Tag & Quick Override Dropdown
  const trPred = document.createElement("tr");
  const thCorner1 = document.createElement("th");
  thCorner1.className = "row-num-col";
  thCorner1.textContent = "Model";
  trPred.appendChild(thCorner1);

  colsMeta.forEach((col) => {
    const isSymCol =
      col.current_header.toLowerCase() === "symptoms" ||
      col.predicted_label === "Symptoms" ||
      col.current_header.toLowerCase().startsWith("symptom:");

    const th = document.createElement("th");
    th.className = `th-prediction-cell ${isSymCol ? "is-symptom-col" : ""}`;

    const card = document.createElement("div");
    card.className = "pred-card";

    // Top row: Route tag + confidence
    const topRow = document.createElement("div");
    topRow.className = "pred-top-row";

    const routeSpan = document.createElement("span");
    let routeClass = "route-tag";
    if (String(col.routed_to).includes("RECA")) routeClass += " route-reca";
    else if (String(col.routed_to).includes("Random Forest") || String(col.routed_to).includes("RF")) routeClass += " route-rf";
    else if (String(col.routed_to).includes("Hybrid") || String(col.routed_to).includes("LLM")) routeClass += " route-hybrid";
    else if (String(col.routed_to).includes("Symptom")) routeClass += " route-symptom";
    routeSpan.className = routeClass;
    routeSpan.textContent = col.routed_to || "Local";
    if (col.triggers && col.triggers.length > 0) {
      routeSpan.title = `Hybrid Routing Triggers: ${col.triggers.join(", ")}`;
    }
    topRow.appendChild(routeSpan);

    if (col.confidence !== null && col.confidence !== undefined) {
      const confSpan = document.createElement("span");
      confSpan.className = "conf-pill";
      const pct = Math.round(col.confidence * 100);
      const diffTxt = col.num_different_answers ? ` · N_diff=${col.num_different_answers}` : "";
      confSpan.textContent = `${pct}%${diffTxt}`;
      confSpan.title = `Confidence: ${(col.confidence * 100).toFixed(1)}% | Distinct Ensemble Votes: ${col.num_different_answers || 1}`;
      topRow.appendChild(confSpan);
    }
    card.appendChild(topRow);

    // Clickable Prediction Badge (Left-click: accept prediction; Right-click: revert to original header)
    const predBtn = document.createElement("button");
    const isAccepted = col.current_header === col.predicted_label;
    const isOverridden = col.status === "overridden" && !isAccepted;
    predBtn.className = `pred-badge-btn ${isAccepted ? "accepted" : ""} ${isOverridden ? "overridden" : ""}`;
    predBtn.title = "Left-click to apply predicted label to column header. Right-click to reset to original CSV header.";
    predBtn.innerHTML = `
      <span>${isAccepted ? "&#10003; " : "&#10140; "}${escapeHtml(col.predicted_label)}</span>
      <span style="font-size:0.7rem;opacity:0.75">${isAccepted ? "Active" : "Apply"}</span>
    `;
    predBtn.addEventListener("click", () => {
      overrideColumnHeader(col.col_index, col.predicted_label, "accepted");
    });
    predBtn.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      overrideColumnHeader(col.col_index, col.original_header, "original");
    });
    card.appendChild(predBtn);

    // Quick Override Select (Top alternatives + All 73 WHO Semantic Classes)
    const sel = document.createElement("select");
    sel.className = "override-select";
    sel.title = "Override prediction with another WHO semantic class";

    const placeholderOpt = document.createElement("option");
    placeholderOpt.value = "";
    placeholderOpt.textContent = "Override class...";
    sel.appendChild(placeholderOpt);

    if (col.alternatives && col.alternatives.length > 0) {
      const altGroup = document.createElement("optgroup");
      altGroup.label = "Top Model Candidates";
      col.alternatives.forEach((alt) => {
        const opt = document.createElement("option");
        opt.value = alt.label;
        opt.textContent = `${alt.label} (${Math.round(alt.prob * 100)}%)`;
        altGroup.appendChild(opt);
      });
      sel.appendChild(altGroup);
    }

    const allGroup = document.createElement("optgroup");
    allGroup.label = "All 73 WHO Semantic Classes";
    APP_STATE.allClasses.forEach((cName) => {
      const opt = document.createElement("option");
      opt.value = cName;
      opt.textContent = cName;
      allGroup.appendChild(opt);
    });
    sel.appendChild(allGroup);

    sel.addEventListener("change", () => {
      if (sel.value) {
        overrideColumnHeader(col.col_index, sel.value, "overridden");
      }
    });
    card.appendChild(sel);

    th.appendChild(card);
    trPred.appendChild(th);
  });
  thead.appendChild(trPred);

  // Row 2: Active Editable Column Header + Symptom Split / Remove Column buttons
  const trHeader = document.createElement("tr");
  const thCorner2 = document.createElement("th");
  thCorner2.className = "row-num-col";
  thCorner2.textContent = "#";
  trHeader.appendChild(thCorner2);

  colsMeta.forEach((col) => {
    const isSymTrigger =
      col.current_header.toLowerCase() === "symptoms" ||
      col.predicted_label === "Symptoms";
    const isGeneratedSymptom = col.current_header.toLowerCase().startsWith("symptom:");

    const th = document.createElement("th");
    th.className = `th-active-header ${isSymTrigger || isGeneratedSymptom ? "is-symptom-col" : ""}`;

    const wrap = document.createElement("div");
    wrap.className = "header-input-wrap";

    const inp = document.createElement("input");
    inp.type = "text";
    inp.className = "header-input";
    inp.value = col.current_header;
    inp.setAttribute("list", "who-semantic-classes");
    inp.title = "Active column header (type any custom name or WHO class and press Enter)";

    const commitInput = () => {
      const val = inp.value.trim();
      if (val && val !== col.current_header) {
        overrideColumnHeader(col.col_index, val, "overridden");
      } else {
        inp.value = col.current_header;
      }
    };
    inp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        inp.blur();
      }
    });
    inp.addEventListener("blur", commitInput);
    wrap.appendChild(inp);

    if (isSymTrigger) {
      const splitBtn = document.createElement("button");
      splitBtn.className = "col-symptom-split-btn";
      splitBtn.title = "Extract boolean symptom columns from this Symptoms column";
      splitBtn.innerHTML = "&#128138; Extract";
      splitBtn.addEventListener("click", () => openSymptomExtractionModal(col.col_index));
      wrap.appendChild(splitBtn);
    }

    const delBtn = document.createElement("button");
    delBtn.className = "col-action-btn";
    delBtn.title = `Remove column "${col.current_header}" (Revertible with Ctrl+Z)`;
    delBtn.innerHTML = "&times;";
    delBtn.addEventListener("click", () => deleteColumn(col.col_index));
    wrap.appendChild(delBtn);

    th.appendChild(wrap);

    const origCap = document.createElement("div");
    origCap.className = "orig-caption";
    origCap.textContent = `Orig: ${col.original_header}`;
    origCap.title = `Original CSV Header: ${col.original_header}`;
    th.appendChild(origCap);

    trHeader.appendChild(th);
  });
  thead.appendChild(trHeader);

  // Data Preview Rows (up to 50 rows)
  (session.preview_rows || []).forEach((rowVals, rIdx) => {
    const tr = document.createElement("tr");
    const tdNum = document.createElement("td");
    tdNum.className = "row-num-col";
    tdNum.textContent = rIdx + 1;
    tr.appendChild(tdNum);

    rowVals.forEach((cellVal, cIdx) => {
      const td = document.createElement("td");
      const strVal = String(cellVal ?? "");
      const isSymCol =
        colsMeta[cIdx] &&
        colsMeta[cIdx].current_header.toLowerCase().startsWith("symptom:");
      td.className = `data-cell ${isSymCol && strVal.toLowerCase() === "true" ? "bool-true" : ""}`;
      td.textContent = strVal.length > 65 ? strVal.slice(0, 65) + "..." : strVal;
      if (strVal.length > 30) td.title = strVal;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

async function overrideColumnHeader(colIndex, newHeader, status = "overridden") {
  try {
    const res = await apiPost("/api/override", {
      col_index: colIndex,
      new_header: newHeader,
      status: status,
    });
    renderSession(res.session, false);
    showToast(`Updated column #${colIndex + 1} to "${newHeader}"`);
    // If user just set a column to "Symptoms", prompt them in the banner
    if (newHeader.trim().toLowerCase() === "symptoms") {
      showToast(`"Symptoms" column detected! Click "Extract" to split into symptom columns.`);
    }
  } catch (err) {
    showToast(`Error: ${err.message}`);
  }
}

async function deleteColumn(colIndex) {
  try {
    const res = await apiPost("/api/delete_column", { col_index: colIndex });
    renderSession(res.session, false);
    showToast("Column removed (Press Ctrl+Z to undo)");
  } catch (err) {
    showToast(`Error: ${err.message}`);
  }
}

async function triggerUndo() {
  try {
    const res = await apiPost("/api/undo");
    if (res.ok) {
      renderSession(res.session, false);
      showToast(res.message || "Reverted last action (Ctrl+Z)");
    } else {
      showToast(res.message || "Nothing to undo");
    }
  } catch (err) {
    showToast(`Undo failed: ${err.message}`);
  }
}

async function triggerRedo() {
  try {
    const res = await apiPost("/api/redo");
    if (res.ok) {
      renderSession(res.session, false);
      showToast(res.message || "Restored action (Ctrl+Y)");
    }
  } catch (err) {
    showToast(`Redo failed: ${err.message}`);
  }
}

// Symptom Extraction Modal Flow
async function openSymptomExtractionModal(colIndex) {
  try {
    showLoading(
      "Running Multilingual Symptom Extraction Pipeline...",
      "Detecting language, running BIOES Medical NER, SpaCy lemmatization, and Fuzzy Clustering (>85%)"
    );
    const res = await apiPost("/api/symptoms/preview", { col_index: colIndex });
    hideLoading();

    APP_STATE.symptomPreview = {
      colIndex: res.col_index,
      targetCol: res.target_col,
      detectedLanguage: res.detected_language,
      items: (res.generated_columns || []).map((c) => ({
        name: c.name,
        true_count: c.true_count,
        total_rows: c.total_rows,
        selected: true,
        isCustom: false,
      })),
    };

    document.getElementById("symptom-modal-subtitle").innerHTML =
      `Source Column: <strong>${escapeHtml(res.target_col)}</strong> &bull; Detected Language: <code>${escapeHtml(res.detected_language)}</code> &bull; ` +
      `Found <strong>${APP_STATE.symptomPreview.items.length}</strong> normalized symptom columns. Uncheck any column to exclude it, or add custom symptoms below.`;

    renderSymptomModalChips();
    document.getElementById("symptom-modal").classList.remove("hidden");
  } catch (err) {
    hideLoading();
    showToast(`Symptom Extraction Error: ${err.message}`);
  }
}

function renderSymptomModalChips() {
  const container = document.getElementById("symptom-chips-container");
  container.innerHTML = "";

  const items = APP_STATE.symptomPreview.items;
  const selectedCount = items.filter((i) => i.selected).length;
  document.getElementById("symptom-count-summary").textContent =
    `${selectedCount} of ${items.length} symptom columns selected`;

  if (items.length === 0) {
    container.innerHTML = `<div style="grid-column:1/-1;text-align:center;color:#64748b;padding:1.5rem;">No symptoms automatically extracted. Add custom symptom columns above.</div>`;
    return;
  }

  items.forEach((item, idx) => {
    const chip = document.createElement("label");
    chip.className = `symptom-chip ${item.selected ? "selected" : ""}`;
    chip.innerHTML = `
      <span style="display:flex;align-items:center;gap:0.4rem;overflow:hidden;">
        <input type="checkbox" ${item.selected ? "checked" : ""} />
        <span style="text-overflow:ellipsis;overflow:hidden;white-space:nowrap;" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
      </span>
      <span class="symptom-chip-count" title="Positive cases in this column">${item.true_count}</span>
    `;
    const chk = chip.querySelector("input");
    chk.addEventListener("change", () => {
      item.selected = chk.checked;
      renderSymptomModalChips();
    });
    container.appendChild(chip);
  });
}

async function applySymptomModalSelection() {
  const prev = APP_STATE.symptomPreview;
  const selectedGenerated = prev.items
    .filter((i) => i.selected && !i.isCustom)
    .map((i) => i.name);
  const selectedCustom = prev.items
    .filter((i) => i.selected && i.isCustom)
    .map((i) => i.name);
  const dropOrig = document.getElementById("chk-drop-orig-symptoms").checked;

  document.getElementById("symptom-modal").classList.add("hidden");
  try {
    showLoading("Applying Extracted Symptom Columns...", "One-hot encoding boolean symptom flags into dataset");
    const res = await apiPost("/api/symptoms/apply", {
      col_index: prev.colIndex,
      selected_columns: selectedGenerated,
      custom_columns: selectedCustom,
      drop_original: dropOrig,
    });
    hideLoading();
    renderSession(res.session, false);
    showToast(`Added ${selectedGenerated.length + selectedCustom.length} symptom columns (Ctrl+Z to revert)`);
  } catch (err) {
    hideLoading();
    showToast(`Failed to apply symptom columns: ${err.message}`);
  }
}

function openQuickAddSymptomModal() {
  if (!APP_STATE.session || !APP_STATE.session.loaded) return;
  const sel = document.getElementById("select-quick-symptom-source");
  sel.innerHTML = `<option value="">Do not scan — initialize all rows to False (or True below)</option>`;
  (APP_STATE.session.columns_meta || []).forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.col_index;
    opt.textContent = `Scan Column #${c.col_index + 1}: ${c.current_header}`;
    if (c.current_header.toLowerCase().includes("symptom")) {
      opt.selected = true;
    }
    sel.appendChild(opt);
  });
  document.getElementById("input-quick-symptom-name").value = "";
  document.getElementById("add-col-modal").classList.remove("hidden");
  document.getElementById("input-quick-symptom-name").focus();
}

async function confirmQuickAddSymptom() {
  const name = document.getElementById("input-quick-symptom-name").value.trim();
  const srcVal = document.getElementById("select-quick-symptom-source").value;
  const defBool = document.getElementById("chk-quick-symptom-default").checked;
  if (!name) {
    showToast("Please enter a symptom name");
    return;
  }
  document.getElementById("add-col-modal").classList.add("hidden");
  try {
    const res = await apiPost("/api/add_custom_symptom_column", {
      symptom_name: name,
      source_col_index: srcVal !== "" ? parseInt(srcVal, 10) : null,
      default_value: defBool,
    });
    renderSession(res.session, false);
    showToast(`Added symptom column "symptom: ${name.toLowerCase()}"`);
  } catch (err) {
    showToast(`Error: ${err.message}`);
  }
}

// File Upload Handler
async function handleFileUpload(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (e) => {
    const content = e.target.result;
    const params = getSelectedRoutingParams();
    try {
      showLoading(
        `Loading & Annotating "${file.name}"...`,
        "Running column type classification and scanning for Symptoms columns"
      );
      const res = await apiPost("/api/upload", {
        filename: file.name,
        content: content,
        auto_annotate: true,
        ...params,
      });
      hideLoading();
      renderSession(res.session, true);
      showToast(`Annotated "${file.name}"`);
    } catch (err) {
      hideLoading();
      showToast(`Upload failed: ${err.message}`);
    }
  };
  reader.readAsText(file);
}

// DOM Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  initStatus();

  document.getElementById("mode-select").addEventListener("change", syncPresetControls);

  document.getElementById("btn-advanced-toggle").addEventListener("click", () => {
    document.getElementById("advanced-drawer").classList.toggle("hidden");
  });

  // File input upload
  const fileInput = document.getElementById("file-upload");
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
      fileInput.value = "";
    }
  });

  // Drag & drop on dropzone
  const dz = document.getElementById("dropzone");
  if (dz) {
    dz.addEventListener("dragover", (e) => {
      e.preventDefault();
      dz.classList.add("dragover");
    });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
  }

  // Sample dataset loader
  const sampleSelect = document.getElementById("sample-select");
  sampleSelect.addEventListener("change", async () => {
    const sName = sampleSelect.value;
    if (!sName) return;
    const params = getSelectedRoutingParams();
    try {
      showLoading(`Loading WHO Sample "${sName}"...`, "Running ensemble column annotation");
      const res = await apiPost("/api/load_sample", {
        sample_name: sName,
        auto_annotate: true,
        ...params,
      });
      hideLoading();
      renderSession(res.session, true);
      showToast(`Loaded and annotated "${sName}"`);
      sampleSelect.value = "";
    } catch (err) {
      hideLoading();
      showToast(`Error: ${err.message}`);
    }
  });

  // Re-run annotation button
  document.getElementById("btn-run-annotate").addEventListener("click", async () => {
    const params = getSelectedRoutingParams();
    try {
      showLoading("Running Column Type Annotation...", `Mode: ${params.mode} (θ=${params.conf_threshold}, τ=${params.ensemble_tolerance})`);
      const res = await apiPost("/api/annotate", params);
      hideLoading();
      renderSession(res.session, true);
      showToast("Annotation complete");
    } catch (err) {
      hideLoading();
      showToast(`Annotation Error: ${err.message}`);
    }
  });

  // Bulk accept / reset
  document.getElementById("btn-accept-all").addEventListener("click", async () => {
    try {
      const res = await apiPost("/api/bulk_accept", { action: "accept_all" });
      renderSession(res.session, false);
      showToast("Accepted all predicted headers (Ctrl+Z to undo)");
    } catch (err) {
      showToast(`Error: ${err.message}`);
    }
  });

  document.getElementById("btn-reset-all").addEventListener("click", async () => {
    try {
      const res = await apiPost("/api/bulk_accept", { action: "reset_all" });
      renderSession(res.session, false);
      showToast("Reset all headers to original CSV names");
    } catch (err) {
      showToast(`Error: ${err.message}`);
    }
  });

  // Undo / Redo buttons & keyboard shortcuts (Ctrl+Z / Ctrl+Y)
  document.getElementById("btn-undo").addEventListener("click", triggerUndo);
  document.getElementById("btn-redo").addEventListener("click", triggerRedo);

  document.addEventListener("keydown", (e) => {
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.shiftKey) {
      e.preventDefault();
      triggerUndo();
    } else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === "y" || (e.shiftKey && e.key.toLowerCase() === "z"))) {
      e.preventDefault();
      triggerRedo();
    }
  });

  // Export CSV button
  document.getElementById("btn-export").addEventListener("click", () => {
    const fmt = document.getElementById("export-format").value;
    window.location.href = `/api/export?format=${encodeURIComponent(fmt)}`;
  });

  // Symptom Modal buttons
  document.getElementById("btn-open-symptom-modal").addEventListener("click", () => {
    const banner = document.getElementById("symptoms-prompt-banner");
    const colIdx = banner.dataset.colIndex ? parseInt(banner.dataset.colIndex, 10) : null;
    openSymptomExtractionModal(colIdx);
  });

  document.getElementById("btn-close-symptom-modal").addEventListener("click", () => {
    document.getElementById("symptom-modal").classList.add("hidden");
  });
  document.getElementById("btn-cancel-symptom-modal").addEventListener("click", () => {
    document.getElementById("symptom-modal").classList.add("hidden");
  });
  document.getElementById("btn-apply-symptom-modal").addEventListener("click", applySymptomModalSelection);

  document.getElementById("btn-sym-select-all").addEventListener("click", () => {
    APP_STATE.symptomPreview.items.forEach((i) => (i.selected = true));
    renderSymptomModalChips();
  });
  document.getElementById("btn-sym-deselect-all").addEventListener("click", () => {
    APP_STATE.symptomPreview.items.forEach((i) => (i.selected = false));
    renderSymptomModalChips();
  });

  const addModalSymInput = document.getElementById("input-new-symptom-modal");
  const addModalSymBtn = document.getElementById("btn-add-symptom-modal");
  const handleAddSymInModal = () => {
    const raw = addModalSymInput.value.trim();
    if (!raw) return;
    const formatted = raw.toLowerCase().startsWith("symptom:") ? raw.toLowerCase() : `symptom: ${raw.toLowerCase()}`;
    const existing = APP_STATE.symptomPreview.items.find((i) => i.name.toLowerCase() === formatted);
    if (existing) {
      existing.selected = true;
    } else {
      APP_STATE.symptomPreview.items.push({
        name: formatted,
        true_count: "custom",
        total_rows: APP_STATE.session.total_rows || 0,
        selected: true,
        isCustom: true,
      });
    }
    addModalSymInput.value = "";
    renderSymptomModalChips();
  };
  addModalSymBtn.addEventListener("click", handleAddSymInModal);
  addModalSymInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAddSymInModal();
    }
  });

  // Quick Add Symptom Column modal
  document.getElementById("btn-Quick-add-symptom").addEventListener("click", openQuickAddSymptomModal);
  document.getElementById("btn-add-symptom-toolbar").addEventListener("click", openQuickAddSymptomModal);
  document.getElementById("btn-close-add-col-modal").addEventListener("click", () => {
    document.getElementById("add-col-modal").classList.add("hidden");
  });
  document.getElementById("btn-cancel-add-col").addEventListener("click", () => {
    document.getElementById("add-col-modal").classList.add("hidden");
  });
  document.getElementById("btn-confirm-add-col").addEventListener("click", confirmQuickAddSymptom);
});
