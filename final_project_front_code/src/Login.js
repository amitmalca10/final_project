import { h } from 'preact';
import { route } from 'preact-router';
import './Login.css'; // ייבוא קובץ ה-CSS

function Login() {
  const handleLogin = async (e) => {
    e.preventDefault();
    const username = e.target.elements.username.value; //gets data from input
    const password = e.target.elements.password.value; //gets data from input

    try {
      const response = await fetch('http://localhost:8000/sign-in', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password })
      });

      const data = await response.json();
     

      if (response.ok) {
        alert('Login successful!');
        route('/add-smell');
      } else {
        // find right error 
        if (response.status === 400 && data.detail === 'User is not found') {
          alert('User not found. Please check your details.');
        } else if (response.status === 400 && data.detail === 'Incorrect username or password') {
          alert('Incorrect username or password. Please try again.');
        } else {
          alert('Login failed. Please try again.');
        }
      }
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred. Please try again.');
    }
  };

  return (
    h('div', { class: 'login-container' },
      h('div', { class: 'logo-container' },
        h('img', {
          src: 'src/LOGO.png', 
          alt: 'Smelltify Logo',
          class: 'login-logo'
        })
      ),
      h('h1', { class: 'login-title' }, 'Welcome to Smelltify'),
      h('form', { class: 'login-form', onSubmit: handleLogin },
        h('div', { class: 'input-group' },
          h('label', { for: 'username', class: 'login-label' }, 'Username:'),
          h('input', {
            type: 'text',
            id: 'username',
            name: 'username',
            placeholder: 'Enter username',
            class: 'login-input'
          })
        ),
        h('div', { class: 'input-group' },
          h('label', { for: 'password', class: 'login-label' }, 'Password:'),
          h('input', {
            type: 'password',
            id: 'password',
            name: 'password',
            placeholder: 'Enter password',
            class: 'login-input'
          })
        ),
        h('button', {
          type: 'submit',
          class: 'login-button'
        }, 'Login')
      ),
      h('p', { class: 'login-signup' },
        'Not registered yet? ',
        h('a', { href: '/register', class: 'login-signup-link' }, 'Sign up')
      ),
      //  Adding the image in the bottom-right corner
       h('div', { class: 'bottom-right-image' },
        h('img', {
          src: 'src/picture.png', 
          alt: 'Decorative Image',
          class: 'decorative-image',
        })
      )
    )
  );
}

export default Login;
