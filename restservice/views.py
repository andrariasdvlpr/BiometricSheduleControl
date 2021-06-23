from rest_framework.views import APIView
from rest_framework.response import Response
from django.contrib.auth.models import User
from core.models import ActivityLog, HistogramReport,DETReport, ROCReport, ActivityLogState, Suggestions, Profile, Employees
from biometric.utils import dencrypt_image
from datetime import date
from datetime import timedelta
from datetime import datetime
from django.conf import settings
from bsc.permissions import IsOptionsOrIsAuthenticated

import base64
import json
import uuid
import shutil
import os


class GetUserData(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)
    
    def get(self, request,userid):
        user = User.objects.get(id =userid)
        template = user.employees.user_template
        img_B64_template = "null"  
        if template :
            aux_path = template.url
            img_B64_template = dencrypt_image(aux_path.replace("%3A", ":"),user.employees.employee_id)
        
        content ={'employee_id':user.employees.employee_id,
         'first_name': user.first_name,
         'last_name': user.last_name,
         'username' : user.username,
         'email' : user.email,
         'image': img_B64_template,
         'decision_threshold' : user.employees.decision_threshold,
         'registration_date' : user.employees.registration_data}
        return Response(content)

class GetActivityLogData(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)  

    def get(self, request,log_id):
        sc = ActivityLog.objects.get(sc_id =log_id)
        core_users = sc.employee
        user = core_users.user
        type = sc.type
       
        sc_img = sc.sc_photo
        img_B64 = "null"
        if sc_img :
            aux_path = sc_img.url
            img_B64 = dencrypt_image(aux_path.replace("%3A", ":"),core_users.employee_id)
            
        template = core_users.user_template
        img_B64_template = "null"  
        if template :
            aux_path = template.url
            img_B64_template = dencrypt_image(aux_path.replace("%3A", ":"),core_users.employee_id)
        
        content ={'employee_id':core_users.employee_id,
         'first_name': user.first_name,
         'last_name': user.last_name,
         'score' : sc.sc_distance,
         'date_control' : sc.sc_date,
         'template' : img_B64_template,
         'type' : type.sc_type,
         'image': img_B64,
         'location': json.loads(sc.sc_location)}
        return Response(content)

class GetUserActivityLogs(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def get(self, request,user_auth_id,option):
        user =  User.objects.get(id =user_auth_id)
        core_users = user.employees
        state = 1 
        if option == "rejected":
            state = 3
        user_activity_logs = core_users.activitylog_set.filter(sc_state=state)
        activity_logs =[]

        for activity_log in user_activity_logs:
            type = activity_log.type  
            row = {'sc_id' : activity_log.sc_id, 'employee_id': activity_log.employee.employee_id, 'sc_type_id':type.sc_type, 'sc_date': activity_log.sc_date.strftime("%b. %d, %Y, %H:%M %p"), 'sc_distance' : activity_log.sc_distance, 'sc_state': activity_log.sc_state}
            activity_logs.append(row)

        return Response(activity_logs)

class GetHistogramData(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def get(self, request):
        data = HistogramReport.objects.all() 
        histogram_data =[]

        for row_data in data:
            row = {'same_distance' : row_data.same_distance, 'diff_distance': row_data.diff_distance}
            histogram_data.append(row)

        return Response(histogram_data)

class GetRocData(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def get(self, request):
        data = ROCReport.objects.all() 
        roc_data =[]

        for row_data in data:
            row = {'FPR' : row_data.FPR, 'ROC': row_data.ROC,'Line':row_data.line}
            roc_data.append(row)

        return Response(roc_data)

class GetDetData(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def get(self, request):
        data = DETReport.objects.all() 
        det_data =[]

        for row_data in data:
            row = {'FPR_log' : row_data.FPR_log, 'DET_log': row_data.DET_log}
            det_data.append(row)

        return Response(det_data)

class GetUserDataAndroid(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def get(self, request):

        user = User.objects.get(username =request.user.username)
        core_users = user.employees
        template = core_users.user_template
        img_B64_template = "null"  
        if template :
            aux_path = template.url
            img_B64_template = dencrypt_image(aux_path.replace("%3A", ":"),core_users.employee_id)
        
        content ={'employee_id':core_users.employee_id,
         'first_name': user.first_name,
         'last_name': user.last_name,
         'username' : user.username,
         'email' : user.email,
         'image': img_B64_template,
         'registration_date' : core_users.registration_data}
        return Response(content)

class GetUserActivityLogsAndroid(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def get(self, request):

        user = User.objects.get(username= request.user.username)
        core_users = user.employees
        user_activity_logs = core_users.activitylog_set.all()
        activity_logs =[]

        for activity_log in user_activity_logs:
            type = activity_log.type
            state = ActivityLogState.objects.get(sc_state_id= activity_log.sc_state)
            row = {'sc_id' : activity_log.sc_id, 'employee_id': core_users.employee_id, 'sc_type':type.sc_type, 'sc_date': activity_log.sc_date, 'sc_distance' : activity_log.sc_distance, 'sc_state': state.sc_state}
            activity_logs.append(row)

        return Response(activity_logs)

class GetActivityLogDataAndroid(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def get(self, request,log_id):

        sc = ActivityLog.objects.get(sc_id = log_id)
        core_users = sc.employee
        user = core_users.user
       
        sc_img = sc.sc_photo
        img_B64 = "null"
        if sc_img :
            aux_path = sc_img.url
            img_B64 = dencrypt_image(aux_path.replace("%3A", ":"),core_users.employee_id)
       
        if sc.sc_location :
            location_json = json.loads(sc.sc_location)
        else :
            location_json = json.loads('{"latitude":0,"longitude":0}')

        content ={
         'first_name': user.first_name,
         'last_name': user.last_name,
         'image': img_B64,
         'location':location_json}

        return Response(content)

class ChangePasswordAndroid(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def post(self, request):

        user = User.objects.get(username =request.user.username)
        received_json_data=json.loads(request.body)
        pss = received_json_data['password']
        user.set_password(pss)
        user.save()
        content ={'state':"successful"}
        return Response(content)

class ReceiveSuggestions(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def post(self, request):
        
        received_json_data=json.loads(request.body)
        suggText = received_json_data['suggestion_text']
        suggestion = Suggestions.objects.create(suggestion_text=suggText)
        suggestion.save()
        content ={'state':"successful"}
        return Response(content)

class GetProfileData(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def get(self, request,profileId):
        profile = Profile.objects.get(profile_id =profileId)
        items =[]

        for type in profile.activitylogtype_set.all():
            row = {'typeId' : type.sc_type_id, 'typeName': type.sc_type}
            items.append(row)

        body = {'profileId'    : profile.profile_id,
                'profileName'  : profile.profile_name,
                'profileState' : profile.profile_state,
                'items'        : items }
        return Response(body)

class IsUsedUsernameOrEmail(APIView):
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            if 'username' in data:
                user = User.objects.filter(username = data["username"])
                if user :
                    resp = {'result':'exist'}
                else :
                    resp = {'result': 'no-exist'}
                return Response(resp)

            if 'email' in data:
                user = User.objects.filter(email = data["email"])
                if user :
                    resp = {'result':'exist'}
                else :
                    resp = {'result': 'no-exist'}
                return Response(resp)
        except:
            return Response({'result': 'error'})

class RegisterUser(APIView):

    def post(self, request):
        try:
            data = json.loads(request.body)  
            user = User.objects.create_user(data["username"], data["email"], data["password"])
            user.first_name = data["firstname"]
            user.last_name = data["lastname"]
            user.save()
            core_users = Employees.objects.create(company= data['username'],employee_id=str(uuid.uuid4()).replace("-",""),registration_data=date.today(),user= user,)
            core_users.save()
            resp = {'result': 'ok'}
        except Exception as e:
            resp = {'result':'error', 'message':'Ha ocurrido un error inesperado'}
        
        return Response(resp)

class GetDailyInfoCurrentWeek(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def get(self, request,date_str):

        dateObj = datetime.strptime(date_str, '%Y-%m-%d')

        dayWeek = dateObj.date()
        dates = [dayWeek + timedelta(days=i) for i in range(0 - dayWeek.weekday(), 7 - dayWeek.weekday())]
        
        user = User.objects.get(username= request.user.username)
        core_users = user.employees
        daily_info = []
        for day in dates:
            user_activity_logs = core_users.activitylog_set.filter(sc_date__contains=day, sc_state = 1)
            activity_logs =[]

            for activity_log in user_activity_logs:
                type = activity_log.type
                row = {'sc_id' : activity_log.sc_id, 'sc_type':type.sc_type_id, 'sc_date': activity_log.sc_date, 'sc_state': activity_log.sc_state}
                activity_logs.append(row)
        
            day_info = { "day" : day.day ,"log": activity_logs}
            daily_info.append(day_info)

        return Response(daily_info)

class DeleteAccountAndroid(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def delete(self, request):

        try:
            user = User.objects.get(username= request.user.username)
            core_users = user.employees
            sc = core_users.activitylog_set.all()
            sc.delete()
            core_users.delete()
            user.delete()

            #Delete files
            user_tmpl_path = os.path.join(settings.FCPATH, "tmplt\\"+str(core_users.id))
            if os.path.exists(user_tmpl_path):
                shutil.rmtree(user_tmpl_path)
            user_dir_path = os.path.join(settings.FCPATH, "log\\"+str(core_users.id))
            if os.path.exists(user_dir_path):
                shutil.rmtree(user_dir_path)
                
            resp = {'result': 'ok'}
        except Exception as e:
            resp = {'result':'error', 'message':'Ha ocurrido un error inesperado'}

        return Response(resp)