/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images:{
    domains:[
      "127.0.0.1", 
      'files.yande.re',
      "user-images.githubusercontent.com",
      "github.com"
    ]
  },
}

module.exports = nextConfig
