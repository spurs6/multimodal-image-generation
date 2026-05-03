(function () {
    "use strict";

    const API_BASE = window.location.origin;
    let currentMode = "sd21";
    let cnControlnetImageBase64 = null;
    let cnReferenceImageBase64 = null;
    /** @type {Record<string, { label_zh?: string, hint_zh?: string, type?: string }>} */
    let referenceModesMeta = {};

    const LS_AUTOSPEAK = "mm-autospeak";

    function showToast(message, type) {
        type = type || "info";
        let region = document.getElementById("toast-region");
        if (!region) {
            region = document.createElement("div");
            region.id = "toast-region";
            region.setAttribute("aria-live", "polite");
            document.body.appendChild(region);
        }
        const el = document.createElement("div");
        el.className = "toast " + type;
        el.textContent = message;
        region.appendChild(el);
        const ms = type === "error" ? 6000 : 4000;
        setTimeout(function () {
            el.remove();
        }, ms);
    }

    function networkErrorHint(msg) {
        const s = String(msg || "").toLowerCase();
        if (
            s.includes("failed to fetch") ||
            s.includes("networkerror") ||
            s.includes("load failed") ||
            s.includes("network request failed")
        ) {
            return (
                (msg || "网络错误") +
                " — 请确认已运行 run_app.bat，且控制台出现 Application startup complete；" +
                "若 8000 被占用，窗口会打印实际 http 地址（请用该地址打开页面）。"
            );
        }
        return msg;
    }

    function formatHttpError(data, status) {
        if (!data) return "HTTP " + status;
        const d = data.detail != null ? data.detail : data.error;
        if (d == null) return "HTTP " + status;
        if (typeof d === "string") return d;
        if (Array.isArray(d)) {
            return d
                .map(function (x) {
                    return x.msg || JSON.stringify(x);
                })
                .join("; ");
        }
        try {
            return JSON.stringify(d);
        } catch (_) {
            return String(d);
        }
    }

    window.switchMode = function (mode) {
        currentMode = mode;
        const tabSd = document.getElementById("tab-sd21");
        const tabCn = document.getElementById("tab-controlnet");
        const panelSd = document.getElementById("sd21-panel");
        const panelCn = document.getElementById("controlnet-panel");

        tabSd.classList.toggle("active", mode === "sd21");
        tabCn.classList.toggle("active", mode === "controlnet");
        tabSd.setAttribute("aria-selected", mode === "sd21" ? "true" : "false");
        tabCn.setAttribute("aria-selected", mode === "controlnet" ? "true" : "false");

        panelSd.classList.toggle("active", mode === "sd21");
        panelCn.classList.toggle("active", mode === "controlnet");

        tabSd.tabIndex = mode === "sd21" ? 0 : -1;
        tabCn.tabIndex = mode === "controlnet" ? 0 : -1;
    };

    function getPrefix() {
        return currentMode === "sd21" ? "sd21" : "cn";
    }

    function syncReferenceAdvancedControls() {
        const mode = document.getElementById("cn-ref-mode");
        const m = mode ? mode.value : "openpose";
        const cnWrap = document.getElementById("cn-cn-scale-wrap");
        const ipWrap = document.getElementById("cn-ip-wrap");
        const controlTypes = { openpose: 1, lineart: 1, softedge: 1, canny: 1 };
        if (cnWrap) cnWrap.style.display = controlTypes[m] ? "block" : "none";
        if (ipWrap) ipWrap.style.display = m === "ip_adapter" ? "block" : "none";
    }

    window.onReferenceModeChange = function () {
        const modeEl = document.getElementById("cn-ref-mode");
        const hintEl = document.getElementById("cn-ref-hint");
        const m = modeEl ? modeEl.value : "openpose";
        const meta = referenceModesMeta[m];
        if (hintEl) {
            hintEl.textContent = meta && meta.hint_zh ? meta.hint_zh : "请选择模式并上传参考图。";
        }
        syncReferenceAdvancedControls();
        if (cnControlnetImageBase64) {
            cnControlnetImageBase64 = null;
            window.resetCannyUpload(true);
            showToast("已切换参考模式，请重新上传参考图", "info");
        }
    };

    window.resetCannyUpload = function (silent) {
        cnControlnetImageBase64 = null;
        cnReferenceImageBase64 = null;
        const prevRef = document.getElementById("cn-reference-preview");
        const prevCanny = document.getElementById("cn-canny-preview");
        const fileInput = document.getElementById("cn-reference-file");
        const drop = document.getElementById("cn-upload-dropzone");
        const status = document.getElementById("cn-upload-status");
        const resetBtn = document.getElementById("cn-reset-upload");
        if (prevRef) {
            prevRef.src = "";
            prevRef.classList.remove("visible");
        }
        if (prevCanny) {
            prevCanny.src = "";
            prevCanny.classList.remove("visible");
        }
        if (fileInput) fileInput.value = "";
        if (drop) drop.classList.remove("hidden");
        if (status) status.classList.add("hidden");
        if (resetBtn) resetBtn.classList.add("hidden");
        if (!silent) {
            showToast("已清除参考图，可重新上传", "info");
        }
    };

    window.handleReferenceUpload = function (event) {
        const file = event.target.files && event.target.files[0];
        if (!file) return;

        const prefix = getPrefix();
        if (prefix !== "cn") return;

        const reader = new FileReader();
        reader.onload = async function (e) {
            cnReferenceImageBase64 = e.target.result.split(",")[1];
            const prevRef = document.getElementById("cn-reference-preview");
            const prevCanny = document.getElementById("cn-canny-preview");
            if (prevRef) {
                prevRef.src = e.target.result;
                prevRef.classList.add("visible");
            }

            const formData = new FormData();
            formData.append("file", file);
            const modeSel = document.getElementById("cn-ref-mode");
            formData.append("ref_mode", modeSel ? modeSel.value : "openpose");

            try {
                const res = await fetch(API_BASE + "/process-reference", {
                    method: "POST",
                    body: formData,
                });
                let data;
                try {
                    data = await res.json();
                } catch (_) {
                    const text = await res.text();
                    throw new Error(text ? text.slice(0, 200) : "无效响应");
                }
                if (!res.ok) {
                    throw new Error(formatHttpError(data, res.status));
                }

                if (prevCanny && data.control_image) {
                    prevCanny.src = data.control_image;
                    prevCanny.classList.add("visible");
                }
                cnControlnetImageBase64 = data.control_image.split(",")[1];

                const drop = document.getElementById("cn-upload-dropzone");
                const status = document.getElementById("cn-upload-status");
                const resetBtn = document.getElementById("cn-reset-upload");
                if (drop) drop.classList.add("hidden");
                if (status) status.classList.remove("hidden");
                if (resetBtn) resetBtn.classList.remove("hidden");

                showToast("参考图已按当前模式处理，可点击「生成图像」", "success");
            } catch (err) {
                showToast("处理图片失败: " + err.message, "error");
                window.resetCannyUpload(true);
            }
        };
        reader.readAsDataURL(file);
    };

    window.enhancePrompt = async function (mode) {
        const prefix = mode;
        const promptEl = document.getElementById(prefix + "-prompt");
        const prompt = promptEl && promptEl.value.trim();
        if (!prompt) {
            showToast("请先输入提示词", "error");
            return;
        }

        const btn = document.getElementById(prefix + "-enhance-btn");
        const loading = document.getElementById(prefix + "-enhance-loading");
        const section = document.getElementById(prefix + "-enhanced-section");

        btn.disabled = true;
        if (loading) loading.style.display = "block";
        if (section) section.style.display = "none";

        showToast("正在调用 DeepSeek 扩充提示词…", "info");
        const statusEl = document.getElementById(prefix + "-enhance-status");
        var enhanceTick = null;
        if (statusEl) {
            statusEl.textContent = "正在连接 API…";
            var eStep = 0;
            enhanceTick = setInterval(function () {
                eStep += 1;
                if (eStep === 1) {
                    statusEl.textContent = "DeepSeek 正在改写提示词…";
                } else {
                    statusEl.textContent = "仍在等待响应（网络较慢时会较久）…";
                }
            }, 4000);
        }

        try {
            const res = await fetch(API_BASE + "/enhance", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt: prompt }),
            });
            let data;
            try {
                data = await res.json();
            } catch (_) {
                const text = await res.text();
                throw new Error(text ? text.slice(0, 200) : "empty response");
            }
            if (!res.ok) {
                throw new Error(formatHttpError(data, res.status));
            }
            if (!data.enhanced_prompt) {
                throw new Error("后端未返回扩充提示词");
            }
            const ta = document.getElementById(prefix + "-enhanced-prompt");
            if (ta) ta.value = data.enhanced_prompt;
            if (section) section.style.display = "block";
            showToast("提示词已扩充；生成时将优先使用下方扩充内容", "success");
        } catch (err) {
            showToast("扩充失败: " + networkErrorHint(err.message), "error");
        } finally {
            if (enhanceTick) {
                clearInterval(enhanceTick);
            }
            if (statusEl) {
                statusEl.textContent = "正在扩充…";
            }
            btn.disabled = false;
            if (loading) loading.style.display = "none";
        }
    };

    function setIndeterminateLoading(prefix, active) {
        const loading = document.getElementById(prefix + "-generate-loading");
        const bar = document.getElementById(prefix + "-progress-bar");
        if (!loading) return;
        if (active) {
            loading.classList.add("active");
            if (bar) bar.classList.add("indeterminate");
        } else {
            loading.classList.remove("active");
            if (bar) bar.classList.remove("indeterminate");
        }
    }

    function startGenerateStatusTicker(prefix) {
        const textEl = document.getElementById(prefix + "-loading-text");
        if (!textEl) {
            return function () {};
        }
        const orig = textEl.textContent;
        textEl.textContent = "已提交请求，等待服务器响应…";
        var step = 0;
        var msgs = [
            "已提交请求，等待服务器响应…",
            "翻译与扩散推理中（GPU）…",
            "仍在生成（高步数或大分辨率会较久）…",
            "即将完成后处理（描述 / CLIP）…",
        ];
        var tid = setInterval(function () {
            step = Math.min(step + 1, msgs.length - 1);
            textEl.textContent = msgs[step];
        }, 3200);
        return function () {
            clearInterval(tid);
            textEl.textContent = orig;
        };
    }

    window.generateImage = async function (mode) {
        const prefix = mode;
        const promptEl = document.getElementById(prefix + "-prompt");
        const prompt = promptEl && promptEl.value.trim();
        if (!prompt) {
            showToast("请先输入提示词", "error");
            return;
        }

        const btn = document.getElementById(prefix + "-generate-btn");
        const result = document.getElementById(prefix + "-generate-result");
        const placeholder = document.getElementById(prefix + "-placeholder-result");

        btn.disabled = true;
        setIndeterminateLoading(prefix, true);
        var stopGenTicker = startGenerateStatusTicker(prefix);
        var origTitle = document.title;
        document.title = "⏳ 生成中… · " + origTitle.replace(/^⏳ 生成中… · /, "");
        if (result) result.style.display = "none";
        if (placeholder) placeholder.style.display = "none";

        try {
            const enhancedPromptEl = document.getElementById(prefix + "-enhanced-prompt");
            const enhancedPrompt = enhancedPromptEl ? enhancedPromptEl.value.trim() : "";

            let requestBody = {
                prompt: prompt,
                num_inference_steps: parseInt(document.getElementById(prefix + "-steps").value, 10),
                guidance_scale: parseFloat(document.getElementById(prefix + "-guidance").value),
            };
            if (enhancedPrompt) {
                requestBody.enhanced_prompt = enhancedPrompt;
            }

            if (mode === "sd21") {
                const backend = document.getElementById("sd21-backend").value;
                if (backend === "sd21") {
                    requestBody.generation_mode = "sd21";
                } else {
                    requestBody.generation_mode = "sd15";
                    requestBody.sd15_style = backend;
                    const resSel = document.getElementById("sd15-resolution");
                    if (resSel) {
                        requestBody.sd15_resolution = parseInt(resSel.value, 10);
                    }
                }
            }

            let res;
            if (mode === "cn") {
                if (!cnControlnetImageBase64) {
                    showToast("请先在本页上传参考图（并等待处理完成）", "error");
                    btn.disabled = false;
                    setIndeterminateLoading(prefix, false);
                    stopGenTicker();
                    document.title = origTitle;
                    if (placeholder) placeholder.style.display = "flex";
                    return;
                }
                const refMode = document.getElementById("cn-ref-mode")
                    ? document.getElementById("cn-ref-mode").value
                    : "openpose";
                const refBody = {
                    ref_mode: refMode,
                    prompt: prompt,
                    control_image: cnControlnetImageBase64,
                    num_inference_steps: parseInt(document.getElementById(prefix + "-steps").value, 10),
                    guidance_scale: parseFloat(document.getElementById(prefix + "-guidance").value),
                    controlnet_conditioning_scale: parseFloat(
                        document.getElementById("cn-cn-scale").value
                    ),
                    ip_adapter_scale: parseFloat(document.getElementById("cn-ip-scale").value),
                };
                if (enhancedPrompt) {
                    refBody.enhanced_prompt = enhancedPrompt;
                }
                res = await fetch(API_BASE + "/generate-reference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(refBody),
                });
            } else {
                res = await fetch(API_BASE + "/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(requestBody),
                });
            }

            let data;
            try {
                data = await res.json();
            } catch (_) {
                const text = await res.text();
                throw new Error(text ? text.slice(0, 200) : "empty response");
            }
            if (!res.ok) {
                throw new Error(formatHttpError(data, res.status));
            }
            if (!data.image) {
                throw new Error("后端未返回图像数据");
            }

            document.getElementById(prefix + "-result-image").src = data.image;
            document.getElementById(prefix + "-result-caption").textContent = data.caption || "";

            if (data.used_num_steps && data.used_guidance) {
                document.getElementById(prefix + "-steps").value = data.used_num_steps;
                document.getElementById(prefix + "-steps-value").textContent = data.used_num_steps;
                document.getElementById(prefix + "-guidance").value = data.used_guidance;
                document.getElementById(prefix + "-guidance-value").textContent = data.used_guidance;
            }

            const clipEl = document.getElementById(prefix + "-clip-evaluation");
            if (data.clip_evaluation && clipEl) {
                const clip = data.clip_evaluation;
                document.getElementById(prefix + "-clip-quality").textContent =
                    "CLIP 语义相似度: " + (clip.similarity_score * 100).toFixed(1) + "%";
                document.getElementById(prefix + "-clip-similarity").textContent = clip.interpretation || "";
                clipEl.style.display = "block";
            } else if (clipEl) {
                clipEl.style.display = "none";
            }

            // 保存质量数据到全局，供弹窗使用
            window._lastImageQuality = data.image_quality || null;
            window._lastPromptQuality = data.prompt_quality || null;

            if (result) result.style.display = "block";
            if (placeholder) placeholder.style.display = "none";

            const autospeak = document.getElementById("prefs-autospeak");
            if (autospeak && autospeak.checked) {
                speakCaption(prefix);
            }
        } catch (err) {
            showToast("生成失败: " + networkErrorHint(err.message), "error");
            if (placeholder) placeholder.style.display = "flex";
        } finally {
            stopGenTicker();
            document.title = origTitle;
            btn.disabled = false;
            setIndeterminateLoading(prefix, false);
        }
    };

    window.speakCaption = function (mode) {
        const prefix = mode;
        const captionEl = document.getElementById(prefix + "-result-caption");
        const caption = captionEl ? captionEl.textContent : "";
        if (caption) {
            const utterance = new SpeechSynthesisUtterance(caption);
            utterance.lang = "zh-CN";
            utterance.rate = 1.0;
            speechSynthesis.speak(utterance);
        }
    };

    window.downloadResultImage = function (prefix) {
        const img = document.getElementById(prefix + "-result-image");
        if (!img || !img.src) {
            showToast("没有可下载的图片", "error");
            return;
        }
        const a = document.createElement("a");
        a.href = img.src;
        a.download = "multimodal-generated.png";
        a.click();
        showToast("已开始下载", "success");
    };

    window.copyResultCaption = function (prefix) {
        const p = document.getElementById(prefix + "-result-caption");
        const text = p ? p.textContent : "";
        if (!text) {
            showToast("暂无描述可复制", "error");
            return;
        }
        navigator.clipboard.writeText(text).then(
            function () {
                showToast("描述已复制到剪贴板", "success");
            },
            function () {
                showToast("复制失败，请手动选择文字", "error");
            }
        );
    };

    function bindPrefs() {
        const cb = document.getElementById("prefs-autospeak");
        if (!cb) return;
        try {
            const saved = localStorage.getItem(LS_AUTOSPEAK);
            if (saved !== null) {
                cb.checked = saved === "true";
            }
        } catch (_) {}
        cb.addEventListener("change", function () {
            try {
                localStorage.setItem(LS_AUTOSPEAK, cb.checked ? "true" : "false");
            } catch (_) {}
        });
    }

    function bindKeyboard() {
        document.addEventListener("keydown", function (e) {
            if (!(e.ctrlKey || e.metaKey) || e.key !== "Enter") return;
            const t = e.target;
            if (!t || !t.id) return;
            if (t.id === "sd21-prompt" || t.id === "cn-prompt") {
                e.preventDefault();
                const mode = t.id === "sd21-prompt" ? "sd21" : "cn";
                window.generateImage(mode);
            }
        });
    }

    function syncSd15ResolutionVisibility() {
        const b = document.getElementById("sd21-backend");
        const w = document.getElementById("sd15-resolution-wrap");
        if (!b || !w) {
            return;
        }
        w.style.display = b.value === "sd21" ? "none" : "block";
    }

    // ========== 质量评估弹窗功能 ==========
    window.openQualityModal = function (type) {
        const modal = document.getElementById("quality-modal");
        const title = document.getElementById("quality-modal-title");
        const scoreEl = document.getElementById("quality-modal-overall-score");
        const gradeEl = document.getElementById("quality-modal-overall-grade");
        const detailsEl = document.getElementById("quality-modal-details");
        const suggestionsEl = document.getElementById("quality-modal-suggestions");
        const suggestionsListEl = document.getElementById("quality-modal-suggestions-list");

        let data, metricNames, isPrompt = false;
        if (type === "image") {
            data = window._lastImageQuality;
            title.textContent = "图片生成质量评估";
            metricNames = {
                "clip_semantic": "语义一致性",
                "aesthetic": "美学评分",
                "sharpness": "清晰度",
                "color_richness": "色彩丰富度",
                "composition": "构图平衡",
                "noise_level": "噪声水平",
                "contrast": "对比度"
            };
        } else {
            data = window._lastPromptQuality;
            title.textContent = "提示词质量评估";
            metricNames = {
                "information_density": "信息密度",
                "structure": "结构完整性",
                "language_purity": "语言纯净度",
                "specificity": "具体性",
                "style_completeness": "风格完整性",
                "length_appropriateness": "长度合理性",
                "quality_boost": "质量增强"
            };
            isPrompt = true;
        }

        if (!data) {
            showToast("暂无评估数据，请先生成图片", "error");
            return;
        }

        const overall = data.overall || {};
        scoreEl.textContent = (overall.score || 0).toFixed(1);
        gradeEl.textContent = overall.grade || "--";

        // 改进建议（仅提示词）
        if (isPrompt && suggestionsEl && suggestionsListEl) {
            const suggestions = [];
            for (const [key, name] of Object.entries(metricNames)) {
                const m = data[key];
                if (m && m.score !== undefined && m.score < 60) {
                    const issue = m.score < 40 ? "严重不足" : "需要改进";
                    const color = m.score < 40 ? "#ef4444" : "#f59e0b";
                    suggestions.push({ name, score: m.score, issue, color });
                }
            }
            if (suggestions.length > 0) {
                suggestionsEl.style.display = "block";
                suggestionsListEl.innerHTML = suggestions.map(s =>
                    `<div class="suggestion-item" style="border-left-color: ${s.color}">` +
                    `<div class="suggestion-metric">${s.name}</div>` +
                    `<div class="suggestion-issue">${s.issue} (当前: ${s.score.toFixed(1)})</div>` +
                    `</div>`
                ).join("");
            } else {
                suggestionsEl.style.display = "none";
            }
        } else if (suggestionsEl) {
            suggestionsEl.style.display = "none";
        }

        // 详情
        let detailsHtml = "";
        for (const [key, name] of Object.entries(metricNames)) {
            const m = data[key];
            if (m && m.score !== undefined) {
                const scoreVal = typeof m.score === "number" ? m.score.toFixed(1) : m.score;
                const pct = Math.min(m.score, 100);
                let color = "#10b981";
                if (m.score < 40) color = "#ef4444";
                else if (m.score < 60) color = "#f59e0b";
                else if (m.score < 80) color = "#3b82f6";
                detailsHtml +=
                    `<div class="modal-metric-card">` +
                    `<div class="modal-metric-header">` +
                    `<span class="modal-metric-name">${name}</span>` +
                    `<span class="modal-metric-grade" style="color:${color}">${m.grade || ""}</span>` +
                    `</div>` +
                    `<div class="modal-metric-score">${scoreVal}</div>` +
                    `<div class="modal-progress-bar">` +
                    `<div class="modal-progress-fill" style="width:${pct}%;background:${color}"></div>` +
                    `</div>` +
                    `</div>`;
            }
        }
        if (data.enhancement_effect) {
            const ee = data.enhancement_effect;
            detailsHtml +=
                `<div class="modal-metric-card enhancement">` +
                `<div class="modal-metric-header">` +
                `<span class="modal-metric-name">增强效果</span>` +
                `<span class="modal-metric-grade">${ee.grade || ""}</span>` +
                `</div>` +
                `<div class="modal-metric-score">${(ee.score || 0).toFixed(1)}</div>` +
                `<div class="modal-progress-bar">` +
                `<div class="modal-progress-fill" style="width:${Math.min(ee.score || 0, 100)}%;background:var(--accent)"></div>` +
                `</div>` +
                `</div>`;
        }
        detailsEl.innerHTML = detailsHtml;

        modal.style.display = "flex";
        document.body.style.overflow = "hidden";
    };

    window.closeQualityModal = function (e) {
        if (e && e.target !== e.currentTarget) return;
        const modal = document.getElementById("quality-modal");
        if (modal) {
            modal.style.display = "none";
            document.body.style.overflow = "";
        }
    };

    // 评估提示词质量（独立按钮）
    window.evaluatePromptQuality = async function () {
        const prefix = getPrefix();
        const promptEl = document.getElementById(prefix + "-prompt");
        const prompt = promptEl ? promptEl.value.trim() : "";
        if (!prompt) {
            showToast("请先输入提示词", "error");
            return;
        }
        const enhancedPromptEl = document.getElementById(prefix + "-enhanced-prompt");
        const enhancedPrompt = enhancedPromptEl ? enhancedPromptEl.value.trim() : "";

        showToast("正在评估提示词质量…", "info");
        try {
            const res = await fetch(API_BASE + "/evaluate-prompt-quality", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: prompt,
                    enhanced_prompt: enhancedPrompt
                })
            });
            let data;
            try {
                data = await res.json();
            } catch (_) {
                const text = await res.text();
                throw new Error(text ? text.slice(0, 200) : "empty response");
            }
            if (!res.ok) {
                throw new Error(formatHttpError(data, res.status));
            }
            window._lastPromptQuality = data.metrics;
            window.openQualityModal("prompt");
        } catch (err) {
            showToast("评估失败: " + networkErrorHint(err.message), "error");
        }
    };

    document.addEventListener("DOMContentLoaded", function () {
        bindPrefs();
        bindKeyboard();
        document.getElementById("tab-sd21").setAttribute("aria-selected", "true");
        document.getElementById("tab-controlnet").setAttribute("aria-selected", "false");
        const backend = document.getElementById("sd21-backend");
        if (backend) {
            backend.addEventListener("change", syncSd15ResolutionVisibility);
            syncSd15ResolutionVisibility();
        }
        fetch(API_BASE + "/reference-modes")
            .then(function (r) {
                return r.json();
            })
            .then(function (cfg) {
                referenceModesMeta = cfg.modes || {};
                const sel = document.getElementById("cn-ref-mode");
                if (sel && cfg.modes) {
                    sel.innerHTML = "";
                    Object.keys(cfg.modes).forEach(function (key) {
                        const m = cfg.modes[key];
                        const opt = document.createElement("option");
                        opt.value = key;
                        opt.textContent = m.label_zh || key;
                        sel.appendChild(opt);
                    });
                }
                window.onReferenceModeChange();
            })
            .catch(function () {
                const hintEl = document.getElementById("cn-ref-hint");
                if (hintEl) {
                    hintEl.textContent =
                        "无法加载模式说明；仍可使用上方默认选项。OpenPose：人物姿势；Lineart：线稿；SoftEdge：柔和边缘；Canny：硬边缘轮廓；IP-Adapter：风格参考。";
                }
                syncReferenceAdvancedControls();
            });
    });
})();
