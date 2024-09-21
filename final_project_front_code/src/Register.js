import { h } from 'preact';
import { route } from 'preact-router';
import './Register.css'; // Importing the CSS file for styling

function Register() {
  // Function to handle the user registration process
  const handleRegister = async (e) => {
    e.preventDefault(); // Prevent the default form submission behavior
    const username = e.target.elements.username.value; // Get the entered username
    const password = e.target.elements.password.value; // Get the entered password
    const email = e.target.elements.email.value; // Get the entered email

    // Check if all fields are filled
    if (username && password && email) {
      try {
        // Send a POST request to the backend to register the user
        const response = await fetch('http://localhost:8000/sign-up', {
          method: 'POST', // HTTP method
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          },
          // Send the registration data (username, password, email) as JSON
          body: JSON.stringify({ username, password, email })
        });

        const data = await response.json(); // Parse the response data as JSON

        // Check if registration was successful
        if (response.ok) {
          alert('Registration successful!');
          route('/add-smell');
        } else if (response.status === 400 && data.detail === "Username or Email already exists") {
          // If the username already exists, show an alert to the user
          alert('Registration failed: Username already exists. Please choose a different username.');
        } else {
          // If there is another error, show the error detail
          alert(`Registration failed: ${data.detail}`);
        }
      } catch (error) {
        // Handle any errors that occur during the registration process
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
      }
    } else {
      // If not all fields are filled, show an alert to the user
      alert('Please fill out all fields.');
    }
  };

  // Function to navigate back to the login screen
  const handleBackToLogin = () => {
    route('/login'); // Navigate to the login page
  };

  return (
    h('div', { class: 'register-container' },
      // "Back to Login" button in the top-right corner
      h('button', {
        class: 'back-to-login-button',
        onClick: handleBackToLogin
      }, 'Back to Login'),
      
      // Smelltify logo display
      h('div', { class: 'logo-container' },
        h('img', {
          src: 'src/LOGO.png', // Path to the logo image
          alt: 'Smelltify Logo',
          class: 'register-logo'
        })
      ),
      // Registration form title
      h('h1', { class: 'register-title' }, 'Sign Up'),
      
      // Registration form
      h('form', { class: 'register-form', onSubmit: handleRegister },
        // Username input field
        h('div', { class: 'register-input-container' },
          h('label', { for: 'username', class: 'register-label' }, 'Username:'),
          h('input', {
            type: 'text',
            id: 'username',
            name: 'username',
            placeholder: 'Enter your username', // Placeholder for the input
            title: 'Username',
            class: 'register-input'
          })
        ),
        // Password input field
        h('div', { class: 'register-input-container' },
          h('label', { for: 'password', class: 'register-label' }, 'Password:'),
          h('input', {
            type: 'password',
            id: 'password',
            name: 'password',
            placeholder: 'Enter your password', // Placeholder for the input
            title: 'Password',
            class: 'register-input'
          })
        ),
        // Email input field
        h('div', { class: 'register-input-container' },
          h('label', { for: 'email', class: 'register-label' }, 'Email:'),
          h('input', {
            type: 'email',
            id: 'email',
            name: 'email',
            placeholder: 'Enter your email', // Placeholder for the input
            title: 'Email',
            class: 'register-input'
          })
        ),
        // Submit button to register the user
        h('button', {
          type: 'submit',
          class: 'register-button'
        }, 'Sign up')
      )
    )
  );
}

export default Register;
