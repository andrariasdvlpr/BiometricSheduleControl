from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

from core.models import AdminAccounts

# Define an inline admin descriptor for Employee model
# which acts a bit like a singleton
class AdminAccountsInline(admin.StackedInline):
    model = AdminAccounts
    can_delete = False
    verbose_name_plural = 'AdminAccounts'

# Define a new User admin
class UserAdmin(BaseUserAdmin):
    inlines = (AdminAccountsInline,)

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)