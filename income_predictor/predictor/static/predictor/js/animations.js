// Loading screen functionality
document.addEventListener("DOMContentLoaded", () => {
    const loadingScreen = document.getElementById("loading-screen")
  
    if (loadingScreen) {
      // Hide loading screen after page loads
      setTimeout(() => {
        loadingScreen.classList.add("hidden")
        setTimeout(() => {
          loadingScreen.style.display = "none"
        }, 500)
      }, 1000)
    }
  
    // Initialize navbar scroll effect
    const navbar = document.querySelector(".navbar")
    if (navbar) {
      window.addEventListener("scroll", () => {
        if (window.scrollY > 10) {
          navbar.classList.add("navbar-scrolled")
        } else {
          navbar.classList.remove("navbar-scrolled")
        }
      })
    }
  
    // Initialize mobile menu toggle
    const menuToggle = document.getElementById("menu-toggle")
    const mobileMenu = document.getElementById("mobile-menu")
  
    if (menuToggle && mobileMenu) {
      menuToggle.addEventListener("click", () => {
        mobileMenu.classList.toggle("hidden")
        document.body.classList.toggle("overflow-hidden")
      })
    }
  
    // Initialize form validation
    const forms = document.querySelectorAll("form")
  
    forms.forEach((form) => {
      const passwordFields = form.querySelectorAll('input[type="password"]')
      const submitButton = form.querySelector('button[type="submit"]')
  
      if (passwordFields.length > 1) {
        // Password confirmation validation
        passwordFields[1].addEventListener("input", () => {
          if (passwordFields[0].value !== passwordFields[1].value) {
            passwordFields[1].classList.add("border-red-500")
            if (submitButton) submitButton.disabled = true
          } else {
            passwordFields[1].classList.remove("border-red-500")
            if (submitButton) submitButton.disabled = false
          }
        })
      }
    })
  
    // Add animation to elements with data-animate attribute
    const animatedElements = document.querySelectorAll("[data-animate]")
  
    if (animatedElements.length > 0) {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              const animation = entry.target.getAttribute("data-animate")
              entry.target.classList.add(animation)
              observer.unobserve(entry.target)
            }
          })
        },
        { threshold: 0.1 },
      )
  
      animatedElements.forEach((element) => {
        observer.observe(element)
      })
    }
  
    // Initialize tooltips
    const tooltips = document.querySelectorAll("[data-tooltip]")
  
    tooltips.forEach((tooltip) => {
      tooltip.addEventListener("mouseenter", function () {
        const tooltipText = this.getAttribute("data-tooltip")
        const tooltipElement = document.createElement("div")
        tooltipElement.className =
          "absolute z-10 px-3 py-2 text-sm font-medium text-white bg-gray-900 rounded-lg shadow-sm tooltip"
        tooltipElement.textContent = tooltipText
        tooltipElement.style.bottom = "calc(100% + 5px)"
        tooltipElement.style.left = "50%"
        tooltipElement.style.transform = "translateX(-50%)"
        this.style.position = "relative"
        this.appendChild(tooltipElement)
      })
  
      tooltip.addEventListener("mouseleave", function () {
        const tooltipElement = this.querySelector(".tooltip")
        if (tooltipElement) {
          tooltipElement.remove()
        }
      })
    })
  
    // Initialize counters
    const counters = document.querySelectorAll("[data-counter]")
  
    if (counters.length > 0) {
      const counterObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              const target = Number.parseInt(entry.target.getAttribute("data-counter"))
              let count = 0
              const duration = 2000 // 2 seconds
              const interval = Math.floor(duration / target)
  
              const counter = setInterval(() => {
                count++
                entry.target.textContent = count
  
                if (count >= target) {
                  clearInterval(counter)
                }
              }, interval)
  
              counterObserver.unobserve(entry.target)
            }
          })
        },
        { threshold: 0.5 },
      )
  
      counters.forEach((counter) => {
        counterObserver.observe(counter)
      })
    }
  })
  
  // Function to show loading screen when navigating
  function showLoading() {
    const loadingScreen = document.getElementById("loading-screen")
    if (loadingScreen) {
      loadingScreen.style.display = "flex"
      loadingScreen.classList.remove("hidden")
    }
    return true
  }
  
  // Add loading screen to all internal links
  document.addEventListener("DOMContentLoaded", () => {
    const internalLinks = document.querySelectorAll('a[href^="/"]')
  
    internalLinks.forEach((link) => {
      if (!link.getAttribute("target")) {
        link.addEventListener("click", showLoading)
      }
    })
  
    // Add loading to form submissions
    const forms = document.querySelectorAll("form")
    forms.forEach((form) => {
      form.addEventListener("submit", showLoading)
    })
  })
  