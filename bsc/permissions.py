from rest_framework import permissions

class IsOptionsOrIsAuthenticated(permissions.BasePermission):        

    def has_permission(self, request, view):
        if request.method == 'OPTIONS':
            return True

        return bool(request.user and request.user.is_authenticated)