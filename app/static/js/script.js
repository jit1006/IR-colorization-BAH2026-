// app/static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        scenes: [],
        currentScene: 'urban',
        currentColormap: 'thermal', // 'thermal' or 'gray'
        currentSliderMode: 'sr',    // 'sr' or 'color'
        images: null,               // Holds base64 image strings from API
        sliderPosition: 50          // percentage (0-100)
    };

    // DOM Elements
    const sceneSelect = document.getElementById('scene-select');
    const sceneTitle = document.getElementById('scene-title');
    const sceneDesc = document.getElementById('scene-desc');
    const runBtn = document.getElementById('run-btn');
    const btnText = runBtn.querySelector('.btn-text');
    const spinner = runBtn.querySelector('.spinner');
    
    const cmapButtons = document.querySelectorAll('.cmap-btn');
    const sliderTabs = document.querySelectorAll('.slider-tab');
    
    // Slider Elements
    const comparisonSlider = document.getElementById('comparison-slider');
    const sliceLeft = comparisonSlider.querySelector('.slice-left');
    const sliderHandle = comparisonSlider.querySelector('.slider-handle');
    const imgLeft = document.getElementById('img-left');
    const imgRight = document.getElementById('img-right');
    const lblLeft = document.getElementById('lbl-left');
    const lblRight = document.getElementById('lbl-right');
    
    // Metrics Elements
    const metricTime = document.getElementById('metric-time');
    const metricSrBasePsnr = document.getElementById('metric-sr-base-psnr');
    const metricSrModelPsnr = document.getElementById('metric-sr-model-psnr');
    const metricSrBaseSsim = document.getElementById('metric-sr-base-ssim');
    const metricSrModelSsim = document.getElementById('metric-sr-model-ssim');
    const metricColorPsnr = document.getElementById('metric-color-psnr');
    const metricColorSsim = document.getElementById('metric-color-ssim');
    const modelStatusText = document.getElementById('model-status-text');
    
    // Grid Elements
    const gridLrTir = document.getElementById('grid-lr-tir');
    const gridSrTir = document.getElementById('grid-sr-tir');
    const gridPredRgb = document.getElementById('grid-pred-rgb');
    const gridGtRgb = document.getElementById('grid-gt-rgb');

    // 1. Fetch available scenes and initialize dropdown
    fetch('/api/scenes')
        .then(res => res.json())
        .then(data => {
            state.scenes = data;
            sceneSelect.innerHTML = '';
            data.forEach(scene => {
                const opt = document.createElement('option');
                opt.value = scene.id;
                opt.textContent = scene.name;
                sceneSelect.appendChild(opt);
            });
            
            // Set initial scene description
            updateSceneInfo(data[0].id);
            
            // Auto-run inference on startup for initial render
            triggerInference();
        })
        .catch(err => console.error("Error loading scenes:", err));

    // Dropdown change handler
    sceneSelect.addEventListener('change', (e) => {
        state.currentScene = e.target.value;
        updateSceneInfo(state.currentScene);
    });

    function updateSceneInfo(sceneId) {
        const scene = state.scenes.find(s => s.id === sceneId);
        if (scene) {
            sceneTitle.textContent = scene.name;
            sceneDesc.textContent = scene.description;
        }
    }

    // 2. Colormap Toggle Buttons
    cmapButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            cmapButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentColormap = btn.getAttribute('data-cmap');
            updateImages();
        });
    });

    // 3. Slider Tab Switchers (SR vs Color)
    sliderTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            sliderTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.currentSliderMode = tab.getAttribute('data-slider-mode');
            updateImages();
        });
    });

    // 4. Interactive Slider Drag Handler
    let isDragging = false;

    function moveSlider(x) {
        const rect = comparisonSlider.getBoundingClientRect();
        let posX = x - rect.left;
        let percentage = (posX / rect.width) * 100;
        
        // Clamp percentage between 2% and 98%
        percentage = Math.max(2, Math.min(98, percentage));
        state.sliderPosition = percentage;
        
        sliceLeft.style.width = `${percentage}%`;
        sliderHandle.style.left = `${percentage}%`;
    }

    // Mouse events
    comparisonSlider.addEventListener('mousedown', (e) => {
        e.preventDefault(); // Prevent default text selection/image drag
        isDragging = true;
        moveSlider(e.clientX);
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        moveSlider(e.clientX);
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
    });

    // Touch events for mobile support
    comparisonSlider.addEventListener('touchstart', (e) => {
        isDragging = true;
        moveSlider(e.touches[0].clientX);
    });

    window.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        moveSlider(e.touches[0].clientX);
    });

    window.addEventListener('touchend', () => {
        isDragging = false;
    });

    // 5. Run Inference Action
    runBtn.addEventListener('click', triggerInference);

    function triggerInference() {
        // UI states
        runBtn.disabled = true;
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        
        // Flow diagram highlighting
        const steps = ['flow-step-1', 'flow-step-2', 'flow-step-3'];
        steps.forEach((stepId, idx) => {
            const stepEl = document.getElementById(stepId);
            stepEl.classList.remove('active');
            setTimeout(() => {
                stepEl.classList.add('active');
            }, idx * 400); // Simulate processing steps
        });

        fetch('/api/inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scene: state.currentScene })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            
            state.images = data;
            
            // Update models status badge
            if (data.using_trained_weights) {
                modelStatusText.textContent = "Pipeline Active (Trained Weights)";
                modelStatusText.parentElement.querySelector('.status-dot').className = "status-dot green";
            } else {
                modelStatusText.textContent = "Preview Mode (Weights Loading...)";
                modelStatusText.parentElement.querySelector('.status-dot').className = "status-dot orange";
            }

            // Render view components
            updateImages();
            updateMetrics(data.metrics);
            updateFlowGrid();
        })
        .catch(err => {
            console.error("Error running inference:", err);
            alert("Inference execution failed. Check console and server logs.");
        })
        .finally(() => {
            runBtn.disabled = false;
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        });
    }

    // 6. Update Visual Elements based on state
    function updateImages() {
        if (!state.images) return;
        
        const cmap = state.currentColormap;
        
        if (state.currentSliderMode === 'sr') {
            // Super Resolution View: Bilinear Baseline vs Our SRNet
            if (cmap === 'thermal') {
                imgLeft.src = `data:image/png;base64,${state.images.base_tir_thermal}`;
                imgRight.src = `data:image/png;base64,${state.images.pred_tir_thermal}`;
            } else {
                imgLeft.src = `data:image/png;base64,${state.images.base_tir_gray}`;
                imgRight.src = `data:image/png;base64,${state.images.pred_tir_gray}`;
            }
            lblLeft.textContent = "Bilinear Baseline (100m)";
            lblRight.textContent = "SRNet Enhanced (100m)";
            lblRight.className = "image-label label-right color-green";
            lblRight.style.color = "var(--color-accent-green)";
        } else {
            // Colorization View: SR TIR vs Colorized RGB
            if (cmap === 'thermal') {
                imgLeft.src = `data:image/png;base64,${state.images.pred_tir_thermal}`;
            } else {
                imgLeft.src = `data:image/png;base64,${state.images.pred_tir_gray}`;
            }
            imgRight.src = `data:image/png;base64,${state.images.pred_rgb}`;
            
            lblLeft.textContent = "Sharpened TIR (100m)";
            lblRight.textContent = "Colorized RGB Composite";
            lblRight.className = "image-label label-right color-blue";
            lblRight.style.color = "var(--color-accent-blue)";
        }
    }

    function updateMetrics(metrics) {
        // Animate metrics change
        animateValue(metricTime, metrics.inference_time_ms, 'ms');
        
        metricSrBasePsnr.textContent = `${metrics.sr_base.psnr} dB`;
        metricSrModelPsnr.textContent = `${metrics.sr_model.psnr} dB`;
        
        metricSrBaseSsim.textContent = metrics.sr_base.ssim;
        metricSrModelSsim.textContent = metrics.sr_model.ssim;
        
        metricColorPsnr.textContent = `${metrics.color_model.psnr} dB`;
        metricColorSsim.textContent = metrics.color_model.ssim;
    }

    function updateFlowGrid() {
        if (!state.images) return;
        
        const cmap = state.currentColormap;
        
        if (cmap === 'thermal') {
            gridLrTir.src = `data:image/png;base64,${state.images.lr_tir_thermal}`;
            gridSrTir.src = `data:image/png;base64,${state.images.pred_tir_thermal}`;
        } else {
            gridLrTir.src = `data:image/png;base64,${state.images.lr_tir_gray}`;
            gridSrTir.src = `data:image/png;base64,${state.images.pred_tir_gray}`;
        }
        
        gridPredRgb.src = `data:image/png;base64,${state.images.pred_rgb}`;
        gridGtRgb.src = `data:image/png;base64,${state.images.gt_rgb}`;
    }

    // Utility: Smoothly increment numbers on metric changes
    function animateValue(obj, end, suffix = '') {
        let start = 0;
        let duration = 500; // ms
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = (progress * (end - start) + start).toFixed(1) + suffix;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
