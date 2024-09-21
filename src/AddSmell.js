import { h } from 'preact';
import { useState } from 'preact/hooks';
import { route } from 'preact-router';
import './AddSmell.css'; // Import the CSS file for styling

function AddSmell() {
  // State to hold the selected file
  const [file, setFile] = useState(null);

  // Function to handle file input change
  const handleFileChange = (e) => {
    setFile(e.target.files[0]); // Set the selected file in the state
  };

  // Function to handle form submission
const handleSubmit = async (e) => {
  e.preventDefault(); // Prevent the form from refreshing the page
  alert("It can take a few seconds");

  // Update button text to indicate processing
  const buttonElement = document.getElementById('GoBtn');
  buttonElement.innerText = 'Processing...';

  // Validate the file
  if (file && file.name.endsWith('.csv')) { // Check if the file is a valid CSV
    const formData = new FormData();
    formData.append('file', file); // Append the file to FormData

    try {
      // Send the file to the backend using a POST request
      const response = await fetch('http://localhost:8000/upload-csv', {
        method: 'POST',
        body: formData, // Send the FormData containing the file
      });
      const data_file = await response.json();
      if (response.ok) { 
        if(data_file.messege=='pass'){//checking if the file is in right length 
          alert('File uploaded and processed successfully!');
          route('/smell-details'); // Navigate to the SmellDetails page
       }
        else if(data_file.messege=="no"){// if the file not in right length 
          alert("sorry,data must be in length of 138 bits per smell!")
          buttonElement.innerText = 'Go!';
        }
        else if(data_file.messege=="no_binar"){// if the file not in right length 
          alert("sorry,data must be binar vector! please upload new file")
          buttonElement.innerText = 'Go!';
        }
      }
      else {
        // Restore the button text
        buttonElement.innerText = 'Go!';
        alert('Failed to upload and process the file.');
      }
    } catch (error) {
      // Restore the button text and handle the error
      buttonElement.innerText = 'Go!';
      console.error('Error:', error);
      alert('An error occurred while uploading the file.');
    }
  } else {
    // Restore the button text and show error if file is not valid
    buttonElement.innerText = 'Go!';
    alert('Please upload a valid CSV file.');
  }
};


  // Function to handle logout and redirect to login page
  const handleLogout = () => {
    route('/login'); // Navigate to the login page
  };

  return (
    h('div', { class: 'addsmell-container' },
      // Logo section
      h('div', { class: 'logo-container' },
        h('img', {
          src: 'src/LOGO.png', // Update with your logo path
          alt: 'Smelltify Logo',
          class: 'addsmell-logo'
        })
      ),

      // Logout button
      h('button', {
        class: 'logout-button',
        onClick: handleLogout // Calls handleLogout on click
      }, 'Logout'),

      // Page title
      h('h1', { class: 'addsmell-title' }, 'Add New Smell'),

      // Form for file upload and submission
      h('form', { onSubmit: handleSubmit, class: 'addsmell-form' },

        // Input for file upload
        h('input', {
          type: 'file',
          accept: '.csv', // Accept only CSV files
          onChange: handleFileChange, // Calls handleFileChange when a file is selected
          class: 'addsmell-input',
          title: 'Upload CSV'
        }),

        // Submit button
        h('button', {
          id:'GoBtn',
          type: 'submit',
          class: 'addsmell-button',
          title: 'Submit',
          //onSubmit: handleSubmit(),
        }, 'GO!'),

        // Display the selected file name if a file is chosen
        file && h('p', { class: 'file-name' }, `Selected file: ${file.name}`)
      )
    )
  );
}

export default AddSmell;
