import { Container, Box } from "@mui/material";
export const MainContainer = (props) => {
    const {children} = props; 

    return (
    <Box sx={{ flexGrow:1, borderRadius:3, p:3,justifyContent:"center", alignItems:"center", display:"flex", width:"80%"}}> 
        {children}
    </Box>
    ); 
}