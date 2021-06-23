from django.db import models
from django.contrib.auth.models import User
import uuid

class Employees(models.Model):
    id = models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False,unique=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    company = models.CharField(max_length=100)
    employee_id = models.CharField(max_length=40,unique=True)
    registration_data = models.DateField()
    decision_threshold = models.FloatField(default=0.3000)
    user_template = models.ImageField(upload_to='users/templates', null = True, blank=True)
    
    class Meta:
        db_table = "employees"

class AdminAccounts(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    company = models.CharField(max_length=100)

    class Meta:
        db_table = "admin_user_company"

class Profile(models.Model):
    profile_id = models.AutoField(primary_key=True)
    profile_name = models.CharField(max_length=150,unique=True)
    profile_state = models.BooleanField( default=True)
    user_admin = models.ForeignKey(AdminAccounts, on_delete=models.DO_NOTHING, null=True)

    class Meta:
        db_table = "profiles"

class ActivityLogType(models.Model):
    sc_type_id = models.AutoField (primary_key=True)
    sc_type = models.CharField(max_length=255)
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)

    class Meta:
        db_table = "activity_log_types"

class ActivityLog(models.Model):
    sc_id = models.AutoField (primary_key=True)
    employee = models.ForeignKey(Employees, on_delete=models.DO_NOTHING)
    type = models.ForeignKey(ActivityLogType, on_delete=models.DO_NOTHING)
    sc_date = models.DateTimeField()
    sc_distance = models.FloatField()
    sc_state = models.IntegerField()
    sc_photo = models.ImageField(upload_to='records', null = True)
    sc_location = models.TextField(null = True)

    class Meta:
        db_table = "activity_logs"

class ActivityLogState(models.Model):
    sc_state_id = models.AutoField (primary_key=True)
    sc_state = models.CharField(max_length=255)
    sc_threshold = models.FloatField(null=True)

    class Meta:
        db_table = "activity_log_state"

class HistogramReport(models.Model):
    id = models.AutoField (primary_key=True)
    same_distance = models.FloatField()
    diff_distance = models.FloatField()

    class Meta:
        db_table = "histogram_report"

class ROCReport(models.Model):
    id = models.AutoField (primary_key=True)
    fpr_log = models.FloatField()
    roc_log = models.FloatField()
    line = models.FloatField()

    class Meta:
        db_table = "roc_report"

class DETReport(models.Model):
    id = models.AutoField (primary_key=True)
    fpr_log = models.FloatField()
    det_log = models.FloatField()

    class Meta:
        db_table = "det_report"

class Suggestions(models.Model):
    suggestion_id = models.AutoField (primary_key=True)
    suggestion_text = models.TextField(null = True)

    class Meta:
        db_table = "suggestions"




