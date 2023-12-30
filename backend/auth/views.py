from django.shortcuts import render
# authentication/views.py

from dj_rest_auth.registration.views import SocialLoginView
from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
from allauth.socialaccount.providers.oauth2.client import OAuth2Client

# Create your views here.
class GoogleLogin(SocialLoginView):
    adapter_class = GoogleOAuth2Adapter
    callback_url = "http://127.0.0.1:3000/"
    client_class = OAuth2Client

    # def complete_login(self, request, app, token, response, **kwargs):
    #     print("Response:",response)
    #     print("Request:",request)
    #     print("App:",app)
    #     print("Token:",token)
    #     super(GoogleLogin, self).complete_login( request, app, token, response, **kwargs)
    def post(self, request, *args, **kwargs):
        print(request.data)
        return super(GoogleLogin, self).post( request, *args, **kwargs)
