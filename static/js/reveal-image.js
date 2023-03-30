document.addEventListener('DOMContentLoaded', function () {
    const hoverImageContainers = document.querySelectorAll('.hover-image-container');
    const lerp_factor_desktop = 0.0004;
    const lerp_factor_mobile = 0.2;

    function lerp(start, end, t) {
        // Linear interpolation
        return start * (1 - t) + end * t;
    }
  
    function animateClipPath(container, targetPercentage, lerp_factor) {
        const currentPercentage = parseFloat(container.getAttribute('data-current-percentage') || '100');
        const newPercentage = lerp(currentPercentage, targetPercentage, lerp_factor);

        container.querySelector('.textured-image').style.clipPath = `polygon(0% 0, calc(${newPercentage}% + 1px) 0, calc(${newPercentage}% + 1px) 100%, 0% 100%)`;
        container.setAttribute('data-current-percentage', newPercentage);

        return Math.abs(newPercentage - targetPercentage) > 0.01;
    }

    function updateTargetPercentage(container, event) {
        const rect = container.getBoundingClientRect();
        const offsetX = event.clientX - rect.left;
        const percentage = (offsetX / rect.width) * 100;
        container.setAttribute('data-target-percentage', percentage);
    }

    function hoverAnimation(container, lerp_factor) {
        const targetPercentage = parseFloat(container.getAttribute('data-target-percentage') || '100');
        const shouldContinue = animateClipPath(container, targetPercentage, lerp_factor);
        if (shouldContinue) {
            requestAnimationFrame(() => hoverAnimation(container, lerp_factor));
        }
    }

    hoverImageContainers.forEach(function (container) {
        // Mouse event handlers
        function startHover(event) {
            container.setAttribute('data-mouse-over', 'true');
            updateTargetPercentage(container, event);
            hoverAnimation(container, lerp_factor_desktop);
        }

        function updateHover(event) {
            updateTargetPercentage(container, event);
            if (container.getAttribute('data-mouse-over') === 'true') {
                hoverAnimation(container, lerp_factor_desktop);
            }
        }

        function endHover() {
            if (container.getAttribute('data-mouse-over') === 'true') {
                container.setAttribute('data-target-percentage', '100');
                hoverAnimation(container, lerp_factor_desktop);
            }
            container.setAttribute('data-mouse-over', 'false');
        }

        // Touch event handlers
        function startTouch(event) {
            event.preventDefault();
            container.setAttribute('data-mouse-over', 'true');
            updateTargetPercentage(container, event.touches[0]);
            hoverAnimation(container, lerp_factor_mobile);
        }

        function updateTouch(event) {
            event.preventDefault();
            updateTargetPercentage(container, event.touches[0]);
            hoverAnimation(container, lerp_factor_mobile);
        }

        function endTouch() {
            if (container.getAttribute('data-mouse-over') === 'true') {
                container.setAttribute('data-target-percentage', '100');
                hoverAnimation(container, lerp_factor_mobile);
            }
            container.setAttribute('data-mouse-over', 'false');
        }

        // Mouse event listeners
        container.addEventListener('mouseenter', startHover);
        container.addEventListener('mousemove', updateHover);
        container.addEventListener('mouseleave', endHover);

        // Touch event listeners
        container.addEventListener('touchstart', startTouch);
        container.addEventListener('touchmove', updateTouch);
        container.addEventListener('touchend', endTouch);
        container.addEventListener('touchcancel', endTouch);
    });
});
  