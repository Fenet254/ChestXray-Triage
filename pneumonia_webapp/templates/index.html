<!DOCTYPE html>
<html lang="en">
  <head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>HealthScan - Pneumonia Detection</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<script>
  tailwind.config = {
    darkMode: 'class',
    theme: {
      extend: {
        colors: {
          primary: "#11d4d4",
          "background-light": "#f6f8f8",
          "background-dark": "#102222",
        },
        fontFamily: {
          display: ["Inter"],
        },
        borderRadius: { DEFAULT: "0.5rem", lg: "1rem", xl: "1.5rem", full: "9999px" },
      },
    },
  };
</script>
</head>
<body class="bg-background-light dark:bg-background-dark font-display text-gray-800 dark:text-gray-200 min-h-screen flex flex-col">

<main class="flex-1 flex justify-center py-8 px-4">

{% if not predictions %}
<!-- Input Form -->
<div class="w-full max-w-2xl bg-white dark:bg-background-dark/50 rounded-xl shadow-lg p-8 space-y-6">
  <h2 class="text-center text-3xl font-bold text-gray-900 dark:text-white mb-4">Pneumonia Detection</h2>
  <form method="POST" enctype="multipart/form-data" class="space-y-4">
    <!-- Patient Info -->
    <div>
      <label class="block text-sm font-medium mb-1" for="patient_name">Patient Name</label>
      <input class="w-full p-3 rounded-lg border border-gray-300 bg-background-light dark:bg-background-dark focus:outline-none focus:ring-2 focus:ring-primary" type="text" name="patient_name" id="patient_name" placeholder="Enter patient's full name" required>
    </div>
    <div class="grid grid-cols-2 gap-4">
      <div>
        <label class="block text-sm font-medium mb-1" for="age">Age</label>
        <input class="w-full p-3 rounded-lg border border-gray-300 bg-background-light dark:bg-background-dark focus:outline-none focus:ring-2 focus:ring-primary" type="number" name="patient_age" id="age" placeholder="e.g. 45" required>
      </div>
      <div>
        <label class="block text-sm font-medium mb-1" for="gender">Gender</label>
        <select class="w-full p-3 rounded-lg border border-gray-300 bg-background-light dark:bg-background-dark focus:outline-none focus:ring-2 focus:ring-primary" name="patient_gender" id="gender" required>
          <option value="">Select gender</option>
          <option>Male</option>
          <option>Female</option>
          <option>Other</option>
        </select>
      </div>
    </div>
    <!-- Symptoms -->
    <div>
      <label class="block text-sm font-medium mb-1" for="symptoms">Symptoms</label>
      <textarea class="w-full p-3 rounded-lg border border-gray-300 bg-background-light dark:bg-background-dark focus:outline-none focus:ring-2 focus:ring-primary" id="patient_symptoms" name="patient_symptoms" rows="4" placeholder="Describe patient's symptoms"></textarea>
    </div>
    <!-- Upload Images -->
    <div>
      <h3 class="text-lg font-bold mb-2">Upload X-ray Images</h3>
      <div class="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-4 flex flex-col items-center justify-center space-y-2 cursor-pointer">
        <input class="hidden" type="file" name="xray_images" multiple id="xray_upload" required>
        <label for="xray_upload" class="cursor-pointer bg-primary text-white px-4 py-2 rounded-lg hover:bg-primary/80">Upload Files</label>
      </div>
    </div>
    <!-- Submit Button -->
    <div>
      <button type="submit" class="w-full bg-primary text-white py-3 rounded-lg hover:bg-primary/80 font-semibold">Predict</button>
    </div>
  </form>
</div>
{% else %}
<!-- Results Display -->
<div class="w-full max-w-4xl mx-auto bg-white dark:bg-background-dark/50 rounded-xl shadow-lg p-6 space-y-6">
<h3 class="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Prediction Results</h3>
{% for result in predictions %}
<div class="bg-white dark:bg-background-dark/50 rounded-xl shadow-lg p-4 mb-4 flex flex-col md:flex-row items-start gap-4">
  <!-- Image Preview -->
  <div class="w-full md:w-1/3 h-48 bg-center bg-no-repeat bg-cover rounded-lg" style='background-image: url("{{ url_for("static", filename="uploads/" + result.filename) }}");'></div>
  <!-- Details -->
  <div class="flex-1 space-y-2">
    <h4 class="font-semibold text-lg text-gray-900 dark:text-white">Image {{ loop.index }}</h4>
    <p class="text-sm text-gray-600 dark:text-gray-300">Diagnosis: <span class="font-semibold text-red-600">{{ result.label }}</span></p>
    <!-- Confidence Circle -->
    <div class="flex items-center space-x-4 mt-4">
      <div class="w-24 h-24 relative flex-shrink-0">
        <svg class="h-full w-full" viewBox="0 0 36 36">
          <circle class="stroke-current text-gray-200 dark:text-gray-700" cx="18" cy="18" r="16" stroke-width="2" fill="none"/>
          <circle class="stroke-current text-primary" cx="18" cy="18" r="16" stroke-width="2"
            stroke-dasharray="{{ result.confidence * 100 }}, 100"
            stroke-linecap="round"
            transform="rotate(-90 18 18)" />
        </svg>
        <div class="absolute inset-0 flex items-center justify-center text-xl font-bold text-primary">{{ (result.confidence * 100) | round(2) }}%</div>
      </div>
    </div>
    <!-- Download PDF button -->
    <div class="mt-4">
      <a href="{{ url_for('download_report') }}" class="bg-primary text-white px-4 py-2 rounded-lg hover:bg-primary/80 font-semibold">Download PDF</a>
    </div>
  </div>
</div>
{% endfor %}
<!-- Start New Session Button -->
<div class="mt-4 flex justify-center">
  <a href="{{ url_for('clear_session') }}" class="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 font-semibold">Start New</a>
</div>
</div>
{% endif %}
</main>
</body>
</html>