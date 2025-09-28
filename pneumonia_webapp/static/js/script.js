document.addEventListener("DOMContentLoaded", () => {
  const predictions = window.predictionsData;

  predictions.forEach((result, index) => {
    const canvasId = `chart${index + 1}`;
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: [result.label],
        datasets: [{
          label: "Confidence (%)",
          data: [result.confidence],
          backgroundColor: result.label === "PNEUMONIA" ? "#d9534f" : "#5cb85c",
          borderColor: "#333",
          borderWidth: 1
        }]
      },
      options: {
        scales: {
          y: { beginAtZero: true, max: 100 }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  });
});