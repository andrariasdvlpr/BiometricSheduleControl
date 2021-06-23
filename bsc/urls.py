"""bsc URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from core import views
from restservice import views as RestFul
from biometric import views as biometric
from django.contrib.auth.decorators import login_required
from rest_framework.authtoken.views import obtain_auth_token
from django.contrib.auth import views as auth_views
from django.views.generic import RedirectView

urlpatterns = [
    #path('', RedirectView.as_view(url='bsc/admin/login/')),
    #path('admin/', admin.site.urls),
    #path('privacypolicy/', views.getPrivacyPolicy.as_view(), name='PrivacyPolicy'),
    path('bsc/admin/login/', views.GetLoginTemplate.as_view(), name='hello'),
    path('bsc/admin/login/verify/', views.DoLogin.as_view(), name='verify'),
    path('bsc/admin/logout/', views.DoLogout.as_view(), name='hello'),
    path('bsc/admin/users/<str:option>', login_required(views.UserPanel.as_view(),login_url="/bsc/admin/login/"), name='dashboard'),
    path('bsc/admin/reports/', login_required(views.ReportsPanel.as_view(),login_url="/bsc/admin/login/"), name='hello'),
    path('bsc/admin/controls/<str:option>', login_required(views.ActivityLogsPanel.as_view(),login_url="/bsc/admin/login/"), name='hello'),
    path('bsc/admin/company/', login_required(views.SetupPanel.as_view(),login_url="/bsc/admin/login/"), name='hello'),
    path('bsc/admin/company/<str:option>', login_required(views.SetupPanel.as_view(),login_url="/bsc/admin/login/"), name='hello'),
    path('bsc/rest-user/<int:userid>',RestFul.GetUserData.as_view()),
    path('bsc/rest-control/<int:log_id>',RestFul.GetActivityLogData.as_view()),
    path('bsc/rest-control-user/<int:user_auth_id>/<str:option>',RestFul.GetUserActivityLogs.as_view()),
    path('bsc/rest-histograms-data/',RestFul.GetHistogramData.as_view()),
    path('bsc/rest-roc-data/',RestFul.GetRocData.as_view()),
    path('bsc/rest-det-data/',RestFul.GetDetData.as_view()),
    path('bsc/rest-profile-data/<int:profileId>',RestFul.GetProfileData.as_view()),
    path('bsc/rest-user-android/',RestFul.GetUserDataAndroid.as_view()),
    path('bsc/rest-user-records-android/',RestFul.GetUserActivityLogsAndroid.as_view()),
    path('bsc/rest-record-data-android/<int:log_id>',RestFul.GetActivityLogDataAndroid.as_view()),
    path('bsc/rest-dailyrecords-week-android/<str:date_str>',RestFul.GetDailyInfoCurrentWeek.as_view()),
    path('bsc/rest-user-exist/',RestFul.IsUsedUsernameOrEmail.as_view()),
    path('bsc/rest-register-user/',RestFul.RegisterUser.as_view()),
    path('api-token-auth/', obtain_auth_token, name='api_token_auth'),
    path('rest-auth/', include('rest_auth.urls')),
    path('bsc/rest-passw-change/',RestFul.ChangePasswordAndroid.as_view()),
    path('bsc/rest-user-suggestions/',RestFul.ReceiveSuggestions.as_view()),
    path('bsc/rest-user-delete/',RestFul.DeleteAccountAndroid.as_view()),
    path('smbConnection/<int:type_activity_log>',biometric.DoFacialVerification.as_view()),
    path('smbConnection/<int:type_activity_log>/<int:date_time>',biometric.DoAsynchronFacialVerification.as_view()),
    path('bsc/rest-user-template-android/',biometric.SetUserTemplate.as_view()),
    path('webapp/set-template/',biometric.SetUserTemplateWebapp.as_view()),
    path('webapp/do-verification/<int:type_activity_log>',biometric.DoFacialVerificationWebApp.as_view()),
    path('user/password/reset/', views.PasswordReset.as_view(), name='password_reset'),
    path('user/password/reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='password/password_reset_done.html'), name='password_reset_done'),
    path('user/reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name="password/password_reset_confirm.html"), name='password_reset_confirm'),
    path('user/reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='password/password_reset_complete.html'), name='password_reset_complete'),
]
