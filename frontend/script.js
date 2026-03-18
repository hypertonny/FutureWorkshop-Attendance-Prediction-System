document.addEventListener('DOMContentLoaded', () => {
    // --- 1. SPLASH SCREEN ---
    const splashScreen = document.getElementById('splash-screen');
    setTimeout(() => {
        splashScreen.style.opacity = '0';
        splashScreen.style.visibility = 'hidden';
        setTimeout(() => splashScreen.remove(), 500); // Remove from DOM
    }, 1800);

    // --- 2. NAVIGATION SCROLLSPY ---
    const sections = document.querySelectorAll('.section');
    const navItems = document.querySelectorAll('.nav-item');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (scrollY >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });
        navItems.forEach(li => {
            li.classList.remove('active');
            if (li.getAttribute('href') === `#${current}`) {
                li.classList.add('active');
            }
        });
    });

    // Semantic smooth scrolling
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            document.querySelector(targetId).scrollIntoView({ behavior: 'smooth' });
        });
    });

    // --- 3. ANIMATE NUMBERS ON LOAD ---
    const counters = document.querySelectorAll('.counter');
    const animateCount = (element) => {
        const target = +element.parentElement.parentElement.getAttribute('data-target');
        const isFloat = element.parentElement.parentElement.getAttribute('data-target').includes('.');
        let count = 0;
        const speed = 200; // lower is slower
        const updateCount = () => {
            const inc = target / speed;
            if (count < target) {
                count += inc;
                element.innerText = isFloat ? count.toFixed(1) : Math.ceil(count).toLocaleString();
                setTimeout(updateCount, 10);
            } else {
                element.innerText = isFloat ? target.toFixed(1) : target.toLocaleString();
            }
        };
        updateCount();
    };

    // Use Intersection Observer to animate only when visible
    const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                const countersInEntry = entry.target.querySelectorAll('.counter');
                countersInEntry.forEach(c => animateCount(c));
                obs.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    sections.forEach(sec => observer.observe(sec));

    // --- 4. FORM LOGIC ---
    const durationSlider = document.getElementById('duration-slider');
    const durationVal = document.getElementById('duration-val');
    durationSlider.addEventListener('input', (e) => durationVal.textContent = e.target.value);

    const regSlider = document.getElementById('reg-slider');
    const regVal = document.getElementById('reg-val');
    regSlider.addEventListener('input', (e) => regVal.textContent = e.target.value);

    // Predict Button Simulation
    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const spinner = predictBtn.querySelector('.spinner');
    const resultCard = document.getElementById('result-card');
    const animatedResultNum = document.querySelector('.animated-result-num');
    const ringFill = document.querySelector('.ring-fill');
    const ringText = document.querySelector('.ring-text');

    predictBtn.addEventListener('click', (e) => {
        e.preventDefault();
        // UI changes
        btnText.textContent = 'Predicting...';
        spinner.classList.remove('hidden');
        resultCard.classList.add('hidden');
        resultCard.style.opacity = '0';
        resultCard.style.transform = 'translateY(20px)';
        
        // Reset ring
        ringFill.style.strokeDashoffset = '314';
        ringText.textContent = '0%';

        setTimeout(() => {
            // Restore btn
            btnText.textContent = 'Predict Attendance';
            spinner.classList.add('hidden');
            
            // Show result
            resultCard.classList.remove('hidden');
            resultCard.style.animation = 'none';
            resultCard.offsetHeight; // trigger reflow
            resultCard.style.animation = 'slideUpFade 0.6s ease-out forwards';
            
            // Generate mock result values
            const baseReg = parseInt(regSlider.value);
            const predictedAttendees = Math.floor(baseReg * (0.4 + Math.random() * 0.4));
            const rate = ((predictedAttendees / baseReg) * 100).toFixed(1);
            
            // Animate result number
            let curr = 0;
            const resInterval = setInterval(() => {
                if (curr >= predictedAttendees) {
                    animatedResultNum.textContent = predictedAttendees;
                    clearInterval(resInterval);
                } else {
                    curr += Math.ceil(predictedAttendees / 20);
                    animatedResultNum.textContent = curr;
                }
            }, 30);
            
            // Animate Ring
            setTimeout(() => {
                const offset = 314 - (rate / 100) * 314;
                ringFill.style.strokeDashoffset = offset;
                ringText.textContent = `${rate}%`;
            }, 200);

        }, 1000);
    });

    // --- 5. CHART.JS VISUAL EDA ---
    Chart.defaults.color = '#8b9bb4';
    Chart.defaults.font.family = "'DM Sans', sans-serif";
    Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 22, 40, 0.9)';
    const chartColors = {
        electric: '#4F8EF7', cyan: '#00D4FF', coral: '#FF6B9D',
        purple: '#8e2de2', yellow: '#ffcc00', green: '#00e676',
        red: '#ff4d4f'
    };

    // 1. Topic Attendance (Horizontal Bar)
    const ctxTopic = document.getElementById('chartTopic').getContext('2d');
    const topics = ["Entrepreneurship", "Machine Learning", "Data Science", "AI & Deep Learning", "Creative Coding", "Web Development", "Career Guidance", "Product Management", "Cybersecurity", "Music Production", "UI/UX Design", "Digital Marketing", "Branding & Identity", "Design Thinking", "Cloud Computing", "Sound Design"];
    const topicRates = Array.from({length: 16}, () => Math.floor(Math.random() * 40 + 30)).sort((a,b)=>b-a);
    new Chart(ctxTopic, {
        type: 'bar',
        data: {
            labels: topics,
            datasets: [{
                label: 'Attendance Rate (%)',
                data: topicRates,
                backgroundColor: chartColors.electric,
                borderRadius: 4
            }]
        },
        options: { indexAxis: 'y', responsive: true, plugins: { legend: { display:false } } }
    });

    // 2. Day of Week
    const ctxDay = document.getElementById('chartDay').getContext('2d');
    new Chart(ctxDay, {
        type: 'bar',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Attendance %',
                data: [42, 46, 48, 45, 41, 28, 25],
                backgroundColor: [chartColors.electric, chartColors.electric, chartColors.electric, chartColors.electric, chartColors.electric, chartColors.coral, chartColors.coral],
                borderRadius: 4
            }]
        },
        options: { responsive: true, plugins: { legend: { display:false } } }
    });

    // 3. Monthly Trend (Combo)
    const ctxMonth = document.getElementById('chartMonth').getContext('2d');
    new Chart(ctxMonth, {
        type: 'line',
        data: {
            labels: ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
            datasets: [
                {
                    label: 'Registrations',
                    data: [150, 200, 400, 350, 380, 220, 250, 480, 500],
                    borderColor: chartColors.electric,
                    yAxisID: 'y'
                },
                {
                    label: 'Attended',
                    data: [65, 90, 180, 140, 170, 95, 120, 230, 240],
                    borderColor: chartColors.green,
                    yAxisID: 'y'
                },
                {
                    type: 'bar',
                    label: 'Rate %',
                    data: [43, 45, 45, 40, 44, 43, 48, 47, 48],
                    backgroundColor: 'rgba(255, 204, 0, 0.4)',
                    yAxisID: 'y1',
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: { type: 'linear', position: 'left' },
                y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, min: 0, max: 100 }
            }
        }
    });

    // 4. Speaker Type
    const ctxSpeaker = document.getElementById('chartSpeaker').getContext('2d');
    new Chart(ctxSpeaker, {
        type: 'bar',
        data: {
            labels: ['Industry', 'Alumni', 'Faculty', 'Student'],
            datasets: [{
                label: 'Rate %',
                data: [57.3, 49.8, 39.9, 31.0],
                backgroundColor: [chartColors.cyan, 'rgba(0, 212, 255, 0.7)', 'rgba(0, 212, 255, 0.5)', 'rgba(0, 212, 255, 0.3)'],
                borderRadius: 4
            }]
        },
        options: { indexAxis: 'y', responsive: true, plugins: { legend: { display:false } } }
    });

    // 5. Online vs Offline
    const ctxMode = document.getElementById('chartMode').getContext('2d');
    new Chart(ctxMode, {
        type: 'bar',
        data: {
            labels: ['Rate %', 'Total Count (x100)'],
            datasets: [
                { label: 'Offline', data: [45.2, 2.5], backgroundColor: chartColors.purple, borderRadius: 4 },
                { label: 'Online', data: [41.8, 12.3], backgroundColor: chartColors.green, borderRadius: 4 }
            ]
        },
        options: { responsive: true, plugins: { legend: { position: 'top' } } }
    });

    // 6. Club Activity
    const ctxClub = document.getElementById('chartClub').getContext('2d');
    new Chart(ctxClub, {
        type: 'bar',
        data: {
            labels: ['Low', 'Medium', 'High'],
            datasets: [{
                label: 'Rate %',
                data: [31, 45.4, 55.2],
                backgroundColor: [chartColors.red, chartColors.yellow, chartColors.green],
                borderRadius: 4
            }]
        },
        options: { responsive: true, plugins: { legend: { display:false } } }
    });

    // 7. Exam Proximity
    const ctxExam = document.getElementById('chartExam').getContext('2d');
    new Chart(ctxExam, {
        type: 'bar',
        data: {
            labels: ['Near', 'Moderate', 'Far'],
            datasets: [{
                label: 'Rate %',
                data: [31.1, 39.1, 50.2],
                backgroundColor: [chartColors.red, chartColors.yellow, chartColors.green],
                borderRadius: 4
            }]
        },
        options: { responsive: true, plugins: { legend: { display:false } } }
    });

    // 8. Department
    const ctxDept = document.getElementById('chartDept').getContext('2d');
    new Chart(ctxDept, {
        type: 'bar',
        data: {
            labels: ['School of Business', 'Design', 'Music', 'Technology'],
            datasets: [{
                label: 'Rate %',
                data: [46.5, 44.9, 42.8, 42.6],
                backgroundColor: chartColors.electric,
                borderRadius: 4
            }]
        },
        options: { indexAxis: 'y', responsive: true, plugins: { legend: { display:false } } }
    });

    // 9. Heatmap simulation (Bubble)
    const ctxHeatmap = document.getElementById('chartHeatmap').getContext('2d');
    const heatmapData = [];
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const times = ['Morning', 'Afternoon', 'Evening'];
    for(let x=0; x<days.length; x++) {
        for(let y=0; y<times.length; y++) {
            // Fake logic for afternoons on weekdays
            let val = 30 + Math.random()*15;
            if (y===1 && x<5) val += 15; // Weekday afternoon boost
            if (y===0 && x>4) val -= 15; // Weekend morning penalty
            
            heatmapData.push({
                x: x, y: y, r: val / 1.5, raw: val // r is size based on value
            });
        }
    }
    new Chart(ctxHeatmap, {
        type: 'bubble',
        data: {
            datasets: [{
                label: 'Attendance %',
                data: heatmapData,
                backgroundColor: function(ctx) {
                    const v = ctx.raw ? ctx.raw.raw : 0;
                    if(v > 45) return 'rgba(0, 230, 118, 0.8)'; // green
                    if(v > 35) return 'rgba(255, 204, 0, 0.8)';  // yellow
                    return 'rgba(255, 77, 79, 0.8)';            // red
                }
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { ticks: { callback: (val) => days[val] }, min: -0.5, max: 6.5 },
                y: { ticks: { callback: (val) => times[val] }, min: -0.5, max: 2.5 }
            },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: (ctx) => `Rate: ${ctx.raw.raw.toFixed(1)}%` } }
            }
        }
    });

    // 10. Semester
    const ctxSem = document.getElementById('chartSemester').getContext('2d');
    new Chart(ctxSem, {
        type: 'bar',
        data: {
            labels: ['1', '2', '3', '4', '5', '6', '7', '8'],
            datasets: [{
                label: 'Rate %',
                data: [52, 48, 45, 42, 40, 39, 36, 32],
                backgroundColor: chartColors.cyan,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                annotation: { // Just a visual mock line for 'dashed average'
                    annotations: [{
                        type: 'line', yMin: 44.2, yMax: 44.2, borderColor: 'rgba(255,255,255,0.5)', borderDash: [5,5]
                    }]
                }
            }
        }
    });

    // --- 6. MODEL NERDS CHARTS ---
    // Model Comparison Grouped Bar
    const ctxMC = document.getElementById('chartModelComparison').getContext('2d');
    new Chart(ctxMC, {
        type: 'bar',
        data: {
            labels: ['XGBoost', 'Random Forest', 'Logistic Regression'],
            datasets: [
                { label: 'F1 Score', data: [0.7125, 0.7322, 0.7337], backgroundColor: chartColors.electric, borderRadius: 2 },
                { label: 'AUC-ROC', data: [0.7847, 0.8073, 0.8267], backgroundColor: chartColors.cyan, borderRadius: 2 },
                { label: 'Accuracy', data: [0.6992, 0.7095, 0.7275], backgroundColor: chartColors.coral, borderRadius: 2 }
            ]
        },
        options: { responsive: true, plugins: { legend: { position: 'top' } } }
    });

    // Radar Chart
    const ctxRadar = document.getElementById('chartRadar').getContext('2d');
    new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['F1 Score', 'Accuracy', 'AUC-ROC'],
            datasets: [
                {
                    label: 'LogReg (Winner)', data: [0.73, 0.72, 0.82],
                    borderColor: chartColors.cyan, backgroundColor: 'rgba(0, 212, 255, 0.2)', borderWidth: 2
                },
                {
                    label: 'Random Forest', data: [0.73, 0.70, 0.80],
                    borderColor: chartColors.purple, backgroundColor: 'rgba(142, 45, 226, 0.2)', borderWidth: 2
                },
                {
                    label: 'XGBoost', data: [0.71, 0.69, 0.78],
                    borderColor: chartColors.coral, backgroundColor: 'rgba(255, 107, 157, 0.2)', borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            scales: { r: { grid: { color: 'rgba(255,255,255,0.1)' }, angleLines: { color: 'rgba(255,255,255,0.1)' }, pointLabels: { color: '#fff' } } },
            plugins: { legend: { position: 'bottom' } }
        }
    });

    // Gauge Chart (Using Half Doughnut)
    const ctxGauge = document.getElementById('chartGauge').getContext('2d');
    new Chart(ctxGauge, {
        type: 'doughnut',
        data: {
            labels: ['Threshold Met', 'Remaining'],
            datasets: [{
                data: [42, 58], // 0.42 threshold
                backgroundColor: [chartColors.electric, 'rgba(255,255,255,0.05)'],
                borderWidth: 0,
                cutout: '80%'
            }]
        },
        options: {
            responsive: true,
            circumference: 180,
            rotation: -90,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            }
        },
        plugins: [{
            id: 'textCenter',
            beforeDraw: function(chart) {
                var width = chart.width, height = chart.height, ctx = chart.ctx;
                ctx.restore();
                var fontSize = (height / 100).toFixed(2);
                ctx.font = 'bold ' + fontSize + "em 'Syne'";
                ctx.fillStyle = '#fff';
                ctx.textBaseline = "middle";
                var text = "0.42",
                    textX = Math.round((width - ctx.measureText(text).width) / 2),
                    textY = height - 20; // push up from bottom
                ctx.fillText(text, textX, textY);
                ctx.save();
            }
        }]
    });

    // --- 7. ACCORDIONS ---
    const accordions = document.querySelectorAll('.accordion');
    accordions.forEach(acc => {
        const btn = acc.querySelector('.accordion-btn');
        btn.addEventListener('click', () => {
            // close others
            accordions.forEach(other => {
                if (other !== acc) other.classList.remove('active');
            });
            // toggle current
            acc.classList.toggle('active');
        });
    });

});
