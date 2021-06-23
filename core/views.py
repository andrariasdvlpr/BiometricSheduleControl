from django.shortcuts import render
from django.template import Template, Context
from django.views import View
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect, render

from django.contrib.auth.models import User
from django.core.mail import send_mail, BadHeaderError
from django.http import HttpResponse
from django.contrib.auth.forms import PasswordResetForm
from django.template.loader import render_to_string
from django.db.models.query_utils import Q
from django.utils.http import urlsafe_base64_encode
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes

from django.contrib import messages
from django.core.mail import EmailMultiAlternatives
from django import template

from core.models import Employees, Profile, ActivityLog, ActivityLogType

from datetime import datetime
from datetime import date
import json
from rest_framework.authtoken.models import Token


class GetLoginTemplate(View):

    def get(self, request):

        ctx = {'next': ""}
        if 'next' in request.GET:
            ctx = {'next': "?next="+request.GET['next']}
        return render(request,"login.html",ctx)

class DoLogin(View):

    def post(self, request):
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_superuser:
            redirect_next = '/bsc/admin/users/add'
            if 'next' in request.GET:
                redirect_next = request.GET['next']
            login(request, user)
            token, created = Token.objects.get_or_create(user=user)
            response = redirect(redirect_next)
            response.set_cookie("FC_SESSIONID", token.key)
            return response
        else:
            redirect_next = '/bsc/admin/login/'
            if 'next' in request.GET:
                redirect_next = '/bsc/admin/login/?next='+request.GET['next']
            return redirect(redirect_next)

class DoLogout(View):

    def get(self, request):
        
        logout(request)
        return redirect('/bsc/admin/login/')
    
class UserPanel(View):

    def get(self, request,option):
        users_auth = Employees.objects.filter(company=request.user.adminaccounts.company)
        ctx = {'option':option, 'users' : users_auth, 'user_admin' : request.user}
        template = option+".html"
        return render(request,template,ctx)
    
    def post(self, request,option):
        if option == "add" :
            user = User.objects.create_user(request.POST["username"], request.POST["email"], request.POST["password"])
            user.first_name = request.POST["firstname"]
            user.last_name = request.POST["lastname"]
            user.save()
            core_users = Employees.objects.create(company= request.user.adminaccounts.company,employee_id=request.POST["cardaccount"],registration_data=date.today(),user= user,)
            core_users.save()
        elif option == "update":
            core_users = Employees.objects.get(employee_id= request.POST["cardaccount"])
            core_users.user.email = request.POST["email"]
            core_users.user.username = request.POST["username"]
            core_users.decision_threshold = request.POST["threshold"]
            core_users.user.save()
            core_users.save()
        elif option == "delete":
            core_users = Employees.objects.get(employee_id= request.POST["cardaccount"])
            core_users.user.delete()
            return redirect("/bsc/admin/users/update")
        
        users_auth = Employees.objects.filter(company=request.user.adminaccounts.company)
        ctx = {'option':option, 'users' : users_auth, 'user_admin' : request.user}
        template = option+".html"
        return render(request,template,ctx)

class ReportsPanel(View):

    def get(self, request):
        return render(request,"reports.html")

class PrivacyPolicy(View):

    def get(self, request):
        return render(request, "PoliticaPrivacidad.html")

class ActivityLogsPanel(View):

    def get(self, request,option):
        activityLogs = ActivityLog.objects.filter(employee__company=request.user.adminaccounts.company, sc_state=2)
        core_users = Employees.objects.filter(company=request.user.adminaccounts.company)
        ctx = {'controls' : activityLogs, 'option' : option, 'users' : core_users}
        template = option+".html"        
        return render(request,template,ctx)
    
    def post(self, request,option):
        activityLog = ActivityLog.objects.get(sc_id=request.POST["record_id"])
        if option == "verify":            
            activityLog.sc_state = 1
        elif option == "reject":            
            activityLog.sc_state = 3
        activityLog.save()     
        return redirect("/bsc/admin/controls/unfinished")

class SetupPanel(View):

    def get(self, request):
        defaultProfile = Profile.objects.get(profile_id =1)
        profiles = Profile.objects.filter(user_admin=request.user.adminaccounts)
        ctx ={"defaultProfile": defaultProfile,
        "companyProfiles" : profiles }
        return render(request,"company.html",ctx)

    def post(self, request,option):
        if request.POST["profile-id"] == "1":
            return redirect("/bsc/admin/company/")

        if option == "add" :
            profile_act = True
            if request.POST['state-profile'] == '0':
                profile_act = False

            company_profile = Profile.objects.create(profile_name= request.POST['profile-name'], profile_state=profile_act, user_admin= request.user.adminaccounts)
            company_profile.save()

            aux = json.loads(request.POST['types'])
            items = aux['items']
            for type in items:
                activity_log_type = ActivityLogType.objects.create(sc_type=type['name'], profile=company_profile)
                activity_log_type.save()

        elif option == "update":
            profile_act = True
            if request.POST['state-profile'] == '0':
                profile_act = False

            company_profile = Profile.objects.get(profile_id=request.POST["profile-id"], user_admin=request.user.adminaccounts)
            company_profile.profile_name = request.POST['profile-name']
            company_profile.profile_state = profile_act
            company_profile.save()

            aux = json.loads(request.POST['types'])
            items = aux['items']
            aux_list =[]
            for type in items:
                if type['id'] =="":
                    activity_log_type = ActivityLogType.objects.create(sc_type=type['name'], profile=company_profile)
                    activity_log_type.save()
                    aux_list.append(activity_log_type.sc_type_id)
                else :
                    activity_log_type = ActivityLogType.objects.get(sc_type_id= type['id'])
                    activity_log_type.sc_type = type['name']
                    activity_log_type.save()
                    aux_list.append(activity_log_type.sc_type_id)
            
            list_types = ActivityLogType.objects.filter(profile=company_profile)
            for type in list_types:
                if type.sc_type_id not in aux_list:
                    type.delete()

        elif option == "delete":
            company_profile = Profile.objects.get(profile_id=request.POST["profile-id"], user_admin=request.user.adminaccounts)
            company_profile.delete()


        return redirect("/bsc/admin/company/")

class PasswordReset(View):

    def get(self, request):
        password_reset_form = PasswordResetForm()
        return render(request=request, template_name="password/password_reset.html", context={"password_reset_form":password_reset_form})

    def post(self, request):
        password_reset_form = PasswordResetForm(request.POST)
        if password_reset_form.is_valid():
            data = password_reset_form.cleaned_data['email']
            associated_users = User.objects.filter(Q(email=data))
            if associated_users.exists():
                for user in associated_users:
                    subject = "Recuperación de contraseña"
                    plaintext = template.loader.get_template('password/password_reset.txt')
                    htmltemp = template.loader.get_template('password/template_email.html')
                    c = {
                    "email":user.email,
                    'domain':'mock.com.es',
                    'site_name': 'mock',
                    "uid": urlsafe_base64_encode(force_bytes(user.pk)),
                    "user": user,
                    'token': default_token_generator.make_token(user),
                    'protocol': 'https',
                    }
                    text_content = plaintext.render(c)
                    html_content = htmltemp.render(c)
                    try:
                        msg = EmailMultiAlternatives(subject, text_content, 'no-responder@mock.com.es', [user.email])
                        msg.attach_alternative(html_content, "text/html")
                        msg.send()
                    except BadHeaderError:
                        return HttpResponse('Invalid header found.')
                    return redirect ("/user/password/reset/done/")