import { h } from 'preact';
import { Router } from 'preact-router';
import Login from './Login';
import Register from './Register';
import AddSmell from './AddSmell';
import SmellDetails from './SmellDetails'; // ייבוא רכיב ה-SmellDetails

function App() {
  return (
    h(Router, null,
      h(Login, { path: '/login' }),
      h(Register, { path: '/register' }),
      h(AddSmell, { path: '/add-smell' }),
      h(SmellDetails, { path: '/smell-details' }), // הוספת הנתיב ל-SmellDetails
      h(Login, { default: true })
    )
  );
}

export default App;
