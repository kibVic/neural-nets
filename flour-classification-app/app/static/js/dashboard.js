function updateDashboard() {
    fetch('/get_dashboard_data')
    .then(response => response.json())
    .then(data => {
        // Update images uploaded count
        document.getElementById('images-uploaded').innerText = data.images_uploaded;
        // Update predictions made count
        document.getElementById('predictions-made').innerText = data.predictions_made;
        // Update last prediction
        document.getElementById('last-prediction').innerText = data.last_prediction;
        // Update recent predictions
        updateRecentPredictions(data.recent_predictions);
    })
    .catch(error => console.log('Error fetching data:', error));
}

function updateRecentPredictions(predictions) {
    const recentPredictionsDiv = document.getElementById('recent-predictions');
    recentPredictionsDiv.innerHTML = '';  // Clear existing predictions
    predictions.forEach(prediction => {
        const predictionItem = document.createElement('div');
        predictionItem.classList.add('prediction-item');
        predictionItem.innerHTML = `
            <div class="prediction-card">
                <img src="${prediction.image_url}" alt="${prediction.image_name}" class="prediction-image">
                <h4>${prediction.image_name}</h4>
                <p>Predicted Flour: <strong>${prediction.predicted_flour}</strong></p>
                <p>Probability: <strong>${prediction.probability}%</strong></p>
            </div>
        `;
        recentPredictionsDiv.appendChild(predictionItem);
    });
}

// Poll the server every 5 seconds to update the dashboard
setInterval(updateDashboard, 5000);
