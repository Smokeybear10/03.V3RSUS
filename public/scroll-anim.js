(function() {
    'use strict';

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Honor reduced motion — reveal everything immediately, skip observers.
    if (reduced) {
        document.querySelectorAll('.reveal').forEach(el => el.classList.add('visible'));
        return;
    }

    // Reveal elements as they enter the viewport.
    const io = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                io.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.12,
        rootMargin: '0px 0px -6% 0px',
    });

    const observe = () => {
        document.querySelectorAll('.reveal:not(.visible)').forEach(el => io.observe(el));
    };

    observe();

    // Re-scan when JS later injects .reveal nodes (e.g. predictor results).
    const mo = new MutationObserver(() => observe());
    mo.observe(document.body, { childList: true, subtree: true });

    // Subtle parallax shift on the hero poster noise while scrolled.
    const noiseTarget = document.querySelector('.hero-poster, .poster');
    if (noiseTarget && window.innerWidth > 720) {
        let ticking = false;
        window.addEventListener('scroll', () => {
            if (ticking) return;
            ticking = true;
            requestAnimationFrame(() => {
                const y = Math.min(window.scrollY, 600);
                noiseTarget.style.setProperty('--scroll-shift', (y * 0.12) + 'px');
                ticking = false;
            });
        }, { passive: true });
    }
})();
