document.addEventListener('DOMContentLoaded', function () {
    const hoverImageContainers = document.querySelectorAll('.hover-image-container');

    function lerp(start, end, t) {
        // Linear interpolation
        return start * (1 - t) + end * t;
    }
  
    function animateClipPath(container, targetPercentage) {
        const currentPercentage = parseFloat(container.getAttribute('data-current-percentage') || '100');
        const newPercentage = lerp(currentPercentage, targetPercentage, 0.0002);

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

    function hoverAnimation(container) {
        const targetPercentage = parseFloat(container.getAttribute('data-target-percentage') || '100');
        animateClipPath(container, targetPercentage);
        const shouldContinue = animateClipPath(container, targetPercentage);
        if (shouldContinue) {
            requestAnimationFrame(() => hoverAnimation(container));
        }
        console.log('hoverAnimation');
    }

    hoverImageContainers.forEach(function (container) {
        container.addEventListener('mouseenter', function (event) {
            container.setAttribute('data-mouse-over', 'true');
            updateTargetPercentage(container, event);
            hoverAnimation(container);
        });

        container.addEventListener('mousemove', function (event) {
            updateTargetPercentage(container, event);
            if (container.getAttribute('data-mouse-over') === 'true') {
                hoverAnimation(container);
            }
        });

        container.addEventListener('mouseleave', function () {
            if (container.getAttribute('data-mouse-over') === 'true') {
                container.setAttribute('data-target-percentage', '100');
                hoverAnimation(container);
            }
            container.setAttribute('data-mouse-over', 'false');
        });
    });
});
  