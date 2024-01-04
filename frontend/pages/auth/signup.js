// pages/signup.js

import { useState } from 'react';
import { Paper, Container, TextField, Button, Typography, Box } from '@mui/material';
import axios from 'axios';
import { ENDPOINTS } from '@/endpoints';
import { signIn } from 'next-auth/react';
export default function SignUp() {
  // const [username, setUsername] = useState('');
  // const [password, setPassword] = useState('');

  const handleSignUp = async (e) => {
    event.preventDefault();

    const data = new FormData(e.target); 
    // Handle signup logic here (e.g., API call to register user)

    // Clear form fields after submission
    // setUsername('');
    // setPassword('');
    const signupURL = ENDPOINTS.AUTH.SIGNUP; 
    const response = await axios.post(signupURL,data); 
    console.log(response.data);
    console.log(response.status);  
    if(response.status==201||response.status==200){
      signIn(undefined, { callbackUrl: '/profile' }); 
    }
  };

  return (
    <Container maxWidth="sm" sx={{mt:10,border:0,display:"flex",justifyContent:"center"}}>
      <Paper sx={{p:2,width:"60%"}}>
      <Typography variant="h4" align="center" gutterBottom>
        Sign Up
      </Typography>
      <form onSubmit={handleSignUp}>
        <TextField
          label="Username"
          variant="outlined"
          required
          fullWidth
          margin="normal"
          name='username'
        />
        <TextField
          label="Password"
          variant="outlined"
          required
          fullWidth
          margin="normal"
          type="password"
          name='password1'
        />
        <TextField
          label="Confirm Password"
          variant="outlined"
          required
          fullWidth
          margin="normal"
          type="password"
          name="password2" 
        />
        <Box width="100%"  justifyContent={"center"}>
        <Button sx={{width:"100%",py:1,my:1}} variant='outlined' color="primary" type="submit">
          Sign Up
        </Button>
        <Button sx={{width:"100%",py:1,my:1}} href="/auth/signin/" variant='outlined' color="primary" >
          Sign In
        </Button>
        </Box>
      </form>
      </Paper>
    </Container>
  );
}
