import { MyHeader } from "./header";
import { MainContainer } from "./mainContainer";
import { Box } from "@mui/material";
export const Layout = (props) => {
    const {children} = props; 
    return (
        <Box  sx={{display:"flex",flexDirection:"column",justifyContent:"center",alignItems:"center"}}>
            <MyHeader/>
            <MainContainer>
                {children} 
            </MainContainer>
        </Box> 
    )
}