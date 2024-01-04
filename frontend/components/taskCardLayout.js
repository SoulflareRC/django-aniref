import Card from '@mui/material/Card';
import { CardHeader, CardActions, Box, Typography, Grid } from '@mui/material';
import CardContent from '@mui/material/CardContent';
import AccountCircle from '@mui/icons-material/AccountCircle';
export const TaskCardLayout = (props) =>{
    // const {data} = props; 
    const {id,name,date,children} = props; 
    return (
        <Card sx={{p:2, display:"flex", flexDirection:"column",  borderRadius: 5, width: "90%", height: "90%", overflow: "auto" }} elevation={5}>
        <CardHeader  avatar={
            <AccountCircle />
        } title={name?name:"Untitled Task"}
        subheader={`Created on ${new Date(date).toLocaleDateString()}`}/>

        <CardContent sx={{display:"flex",flexDirection:"column",  flexGrow:1, height: "auto" }}>
            {children}
        </CardContent>
        <CardActions sx={{ border:0, marginTop:"auto"}}>
            <Typography width={"100%"} border={0} textAlign={"right"} px={2} fontSize={"small"} color={"text.secondary"}>
                Task id {id}
            </Typography>
        </CardActions>
        </Card>
    );
};