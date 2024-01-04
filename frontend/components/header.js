import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import {MenuItem,Menu} from '@mui/material';
import Typography from '@mui/material/Typography';
import { Button } from '@mui/material';

import AccountCircle from '@mui/icons-material/AccountCircle';
import LoginIcon from '@mui/icons-material/Login'
import LogoutIcon from "@mui/icons-material/Logout"
import ListAltIcon from "@mui/icons-material/ListAlt" 
import { signIn, signOut, useSession } from 'next-auth/react';
import Link from 'next/link';

export function MyHeader() {

  const menuId = 'primary-search-account-menu';

  const [anchorElNav, setAnchorElNav] = React.useState(null);
  const handleCloseNavMenu = () => {
    setAnchorElNav(null);
  };
  const handleOpenNavMenu = (event) => {
    console.log("Open nav menu"); 
    setAnchorElNav(event.currentTarget);
  };

  const {data: session, status} = useSession();
//   console.log(session); 
  let icon = null; 
  let hint = null; 
  let handler = () => {}; 
  const authenticated = status === "authenticated"; 
  if(authenticated){
    icon = (
        <AccountCircle />
    )
    hint = session.name?session.name:session.user?.username; 
    handler = handleOpenNavMenu; 
  }else if(status=="unauthenticated"){
    icon = (
        <LoginIcon />
    )
    hint = "Sign in"
    handler = ()=>signIn(); 
  }

  const userMenu = (
    <Menu
        id="menu-appbar"
        anchorEl={anchorElNav}
        anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'left',
        }}
        keepMounted
        transformOrigin={{
            vertical: 'top',
            horizontal: 'left',
        }}
        open={Boolean(anchorElNav)}
        onClose={handleCloseNavMenu}

    >   
        <MenuItem sx={{width:"100%",border:0}}
          component={Link}
          href="/profile">
            <ListAltIcon/> <Typography mx={1}>My Tasks</Typography> 
        </MenuItem>
        <MenuItem onClick={signOut}>
            <LogoutIcon/> <Typography mx={1}>Sign out</Typography> 
        </MenuItem>

    </Menu>
  )


  return (
    <Box sx={{ flexGrow: 0, width:"100%" }}>
      <AppBar position="static">
        <Toolbar>
        <Typography variant='h5' component="a" href="/"
        sx={{
          textDecoration:"none",
          color:"inherit", 
          fontWeight: 600, 
          letterSpacing: "0.1em", 
        }}>Aniref</Typography>
        <Box sx={{ flexGrow: 1, display:'flex' }}>
          <Box sx={{ flexGrow: 1 }} /> 
          <Button variant='white' onClick={handler} startIcon={icon} sx={{ display: { md: 'flex' }, justifyContent:"center", alignItems:"center",  border: 0}}>
             {hint}
          </Button>
          {authenticated && userMenu}
        </Box>
        </Toolbar>
      </AppBar>
    </Box>
  );
}