// pages/api/auth/[...nextauth].js

import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import GoogleProvider from "next-auth/providers/google";
import axios from "axios";

// These two values should be a bit less than actual token lifetimes
const BACKEND_ACCESS_TOKEN_LIFETIME = 45 * 60;            // 45 minutes
const BACKEND_REFRESH_TOKEN_LIFETIME = 6 * 24 * 60 * 60;  // 6 days

const getCurrentEpochTime = () => {
  return Math.floor(new Date().getTime() / 1000);
};

// const SIGN_IN_HANDLERS = {
//   "credentials": async (user, account, profile, email, credentials) => {
//     return true;
//   },// credentials provider handler returns true 
// };

// pages/api/auth/[...nextauth].js

const SIGN_IN_HANDLERS = {
    // ...
    "credentials": async (user, account, profile, email, credentials) => {
        return true;
    },// credentials provider handler returns true 
    "google": async (user, account, profile, email, credentials) => {
      try {
        console.log("Google token:",account["id_token"]); 
        const response = await axios.post(
            process.env.NEXTAUTH_BACKEND_URL + "auth/google/",{
            access_token: account["id_token"],  
            id_token: account["id_token"]
          }
        );
        account["meta"] = response.data;
        return true;
      } catch (error) {
        console.error(error);
        return false;
      }
    }
  };

const SIGN_IN_PROVIDERS = Object.keys(SIGN_IN_HANDLERS);

export const authOptions = {
  secret: process.env.NEXTAUTH_SECRET,
  session: {
    strategy: "jwt",
    maxAge: BACKEND_REFRESH_TOKEN_LIFETIME,
  },
  pages: {
    signIn:"/auth/signin"
  }, 
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        username: {label: "Username", type: "text"},
        password: {label: "Password", type: "password"}
      }, // login form 
      // The data returned from this function is passed forward as the
      // `user` variable to the signIn() and jwt() callback
      async authorize(credentials, req) {
        try {
          const response = await axios.post(process.env.NEXTAUTH_BACKEND_URL + "auth/login/",credentials);
          const data = response.data;
          if (data) return data;
        } catch (error) {
          console.error(error);
        }
        return null;
      },// custom credential logic 
    }),
    GoogleProvider({
        clientId: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        authorization: {
          params: {
            prompt: "consent",
            access_type: "offline",
            response_type: "code"
          }
        }
      }),
  ],
  callbacks: {
    // server side functions 
    async signIn({user, account, profile, email, credentials}) {
        // called after successful sign in 
    //   console.log("Callback signIn"); 
    //   console.log(user, account, profile, email, credentials);
      /*
        user: things returned from login request 
        account: contains {
            providerAccountID, type, provider 
        }
        profile: ???
        email: ??? 
        credentials: {
            csrfToken, username, password 
        } 

      */ 
      if (!SIGN_IN_PROVIDERS.includes(account.provider)) return false;
      return SIGN_IN_HANDLERS[account.provider](
        user, account, profile, email, credentials
      );
    }, // called after success signin 
    async jwt({user, token, account}) {
        /*
        called when a jwt is created(at sign in) or updated(at refresh)

        */ 
      console.log("Callback jwt"); 
      // If `user` and `account` are set that means it is a login event
      if (user && account) {
        let backendResponse = account.provider === "credentials" ? user : account.meta;
        token["user"] = backendResponse.user;
        token["access_token"] = backendResponse.access;
        token["refresh_token"] = backendResponse.refresh;
        token["ref"] = getCurrentEpochTime() + BACKEND_ACCESS_TOKEN_LIFETIME;
        return token;
      }
      // Refresh the backend token if necessary
      if (getCurrentEpochTime() > token["ref"]) {
        const response = await axios({
          method: "post",
          url: process.env.NEXTAUTH_BACKEND_URL + "auth/token/refresh/",
          data: {
            refresh: token["refresh_token"],
          },
        });
        token["access_token"] = response.data.access;
        token["refresh_token"] = response.data.refresh;
        token["ref"] = getCurrentEpochTime() + BACKEND_ACCESS_TOKEN_LIFETIME;
      }
      return token;
    },
    // Since we're using Django as the backend we have to pass the JWT
    // token to the client instead of the `session`.
    async session({token}) {
        // called when a session is checked 
      return token;
    },
  }
};

export default NextAuth(authOptions);