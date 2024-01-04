// pages/signin.js

import { signIn, getProviders } from 'next-auth/react';
// import { Container, TextField, Button, Typography } from '@mui/material';
import { Divider,  Paper, Container, TextField, Button, Typography, Box } from '@mui/material';
import { Google } from '@mui/icons-material';
export default function SignIn(props) {
  const {providers} = props; 
  const handleSignIn = async (event) => {
    event.preventDefault();
    
    const username = event.target.username.value;
    const password = event.target.password.value;

    try {
      const result = await signIn('credentials', {
        username,
        password,
        callbackUrl:"/profile"
      });

      if (!result.error) {
        // Redirect to dashboard or another page on successful login
        // router.push('/dashboard');
      } else {
        // Handle login error
        console.error('Login failed:', result.error);
      }
    } catch (error) {
      console.error('Error during sign in:', error);
    }
  };
//   const googleLoginBtn = (
//     <Button startIcon></Button>
//   )
  return (
    <Container maxWidth="sm" sx={{mt:10,border:0,display:"flex",justifyContent:"center"}}>
    <Paper sx={{p:2,width:"60%"}}>
    <Typography variant="h4" align="center" gutterBottom>
      Sign In
    </Typography>
    <form onSubmit={handleSignIn}>
    <Box sx={{py:1}}>
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
        name='password'
      />
      <Box width="100%" justifyContent={"center"}>
        <Button sx={{width:"100%",py:1,my:1}} variant='outlined' color="primary" type="submit">
          Sign In 
        </Button>
        <Button sx={{width:"100%",py:1,my:1}} href='/auth/signup/' variant='outlined' color="primary" >
          Sign Up 
        </Button>
      </Box>
      </Box>
      <Divider/>
      <Box sx={{py:1}}>
        <Button startIcon={<Google/>} onClick={()=>signIn(providers.google.id,{callbackUrl:"/profile"})} sx={{width:"100%",py:1,my:1}} variant='outlined' color="primary">Google</Button>
      </Box>
    </form>
    </Paper>
  </Container>
  );
}

export async function getServerSideProps(context) {
    const providers = await getProviders(); 
    // console.log(providers); 
    const props = {
        providers 
    }
    return {props}
}