import { useState } from 'react';
import { FormLabel, Box, IconButton, Popover, Typography } from '@mui/material'
import HelpOutlineIcon from "@mui/icons-material/HelpOutline" 
export const PopoverHint = (props) => {
    const {hint} = props; 
    const [anchorEl, setAnchorEl] = useState(null);

    const handlePopoverOpen = (event) => {
      setAnchorEl(event.currentTarget);
    };
  
    const handlePopoverClose = () => {
      setAnchorEl(null);
    };

    const open = Boolean(anchorEl);
    return (
        <Box {...props}> 
            <IconButton
                aria-owns={open ? 'video-help-popover' : undefined}
                aria-haspopup="true"
                onClick={handlePopoverOpen}
                // onMouseLeave={handlePopoverClose}
              >
                <HelpOutlineIcon  fontSize='small'/>
              </IconButton>
            <Popover
              id="video-help-popover"
              open={open}
              anchorEl={anchorEl}
              onClose={handlePopoverClose}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'center',
              }}
              
              transformOrigin={{
                vertical: 'top',
                horizontal: 'center',
              }}
            >
              <Box p={2}>
                <Typography color={"text.secondary"} fontSize={"small"} >
                    {hint} 
                </Typography>
              </Box>
            </Popover>
        </Box>
    )
}