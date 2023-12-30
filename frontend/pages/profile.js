// pages/profile.tsx

import {useState} from "react";
import {signOut, useSession} from "next-auth/react";
// import {Box, Button, Code, HStack, Spinner, Text, VStack} from "@chakra-ui/react";
import {Card, CardHeader,Button} from 'react-bootstrap'; 
import axios from "axios";

export default function Home() {

  const {data: session, status} = useSession({required: true});
  const [response, setResponse] = useState("{}");

  const getUserDetails = async (useToken) => {
    try {
      const response = await axios({
        method: "get",
        url: process.env.NEXT_PUBLIC_BACKEND_URL + "auth/user/",
        headers: useToken ? {Authorization: "Bearer " + session?.access_token} : {},
      });
      setResponse(JSON.stringify(response.data));
    } catch (error) {
      setResponse(error.message);
    }
  };

  if (status == "loading") {
    return <div>Loading...</div>
  }

  if (session) {
    return (
      <Card >
        {/* <VStack> */}
          <div>PK: {session.user.pk}</div>
          <div>Username: {session.user.username}</div>
          <div>Email: {session.user.email || "Not provided"}</div>
          {/* <Code>
            {response}
          </Code> */}
        {/* </VStack> */}
        {/* <HStack justifyContent="center" mt={4}> */}
          <Button colorScheme="blue" onClick={() => getUserDetails(true)}>
            User details (with token)
          </Button>
          <Button colorScheme="orange" onClick={() => getUserDetails(false)}>
            User details (without token)
          </Button>
          <Button colorScheme="red" onClick={() => signOut({callbackUrl: "/"})}>
            Sign out
          </Button>
        {/* </HStack> */}
      </Card>
    );
  }

  return <></>;
}