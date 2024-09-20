document.getElementById('addMovieLink').addEventListener('click', function() {
    document.getElementById('addMovieSection').style.display = 'block';
});

document.getElementById('submitMovie').addEventListener('click', function() {
    var movieName = document.getElementById('movieName').value;
    if (movieName) {
        // Here you would typically add the movie to the database
        document.getElementById('confirmationPopup').style.display = 'block';
        // Clear the input field
        document.getElementById('movieName').value = '';
    }
});

document.getElementById('closePopup').addEventListener('click', function() {
    document.getElementById('confirmationPopup').style.display = 'none';
});

// Example Profile page update function
function updateProfilePage(movieName) {
    // Code to add movie to the profile page
}
