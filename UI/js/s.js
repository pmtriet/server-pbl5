function takeAPic() {
    const video = document.getElementById('camera');
    const predictButton = document.getElementById('predict');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.style.display = 'block';
            predictButton.style.display = 'block';
        })
        .catch(error => {
            console.error('Error accessing media devices.', error);
        });
        document.getElementById('upload-form').style.display = 'none';
}

function chooseAnImage() {
    var form = document.getElementById('upload-form');
    var resultDiv = document.getElementById('result');
    form.style.display = 'block'; // Hiển thị form
    resultDiv.innerHTML = ''; // Xóa nội dung của div kết quả (nếu có)
    camera.style.display = 'none';
    document.getElementById('predict').style.display = 'none';
}

document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('file-input');
    formData.append('image', fileInput.files[0]);

    try {
        const response = await fetch('http://10.25.2.12:5001/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        document.getElementById('result').innerText = `Prediction: ${result.label} with probability ${result.probability}`;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerText = `Error: ${error.message}`;
    }
});

function predict() {
}