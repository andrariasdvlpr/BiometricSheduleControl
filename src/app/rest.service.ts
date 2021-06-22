import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of} from 'rxjs';
import { LoginService } from './login.service';

@Injectable({
  providedIn: 'root'
})
export class RestService {

  constructor(private http : HttpClient, private loginService : LoginService) { }

  getActivityLog() : Observable<any> {
    const token = this.loginService.getToken();
    const headers = new HttpHeaders({'authorization': 'Token '+token});
    return this.http.get("http://192.168.0.16/bsc/rest-user-records-android/",{headers :headers});
  }

  getActivityLogData(id : string) :Observable<any> {
    const token = this.loginService.getToken();
    const headers = new HttpHeaders({'authorization': 'Token '+token});
    return this.http.get("http://192.168.0.16/bsc/rest-record-data-android/"+id,{headers :headers});
  }

  getProfileUser() : Observable<any> {
    const token = this.loginService.getToken();
    const headers = new HttpHeaders({'authorization': 'Token '+token});
    return this.http.get("http://192.168.0.16/bsc/rest-user-android/",{headers :headers});
  }

  doFacialVerification(user: any,type: string) : Observable<any> {
    const token = this.loginService.getToken();
    const headers = new HttpHeaders({'authorization': 'Token '+token});
    return this.http.post("http://192.168.0.16/webapp/do-verification/"+type,user,{headers : headers});
  }

  doSuggestion(data: any) : Observable<any> {
    const token = this.loginService.getToken();
    const headers = new HttpHeaders({'authorization': 'Token '+token});
    return this.http.post("http://192.168.0.16/bsc/rest-user-suggestions/",data,{headers : headers});
  }

  setUserTemplate(data: any) : Observable<any> {
    const token = this.loginService.getToken();
    const headers = new HttpHeaders({'authorization': 'Token '+token});
    return this.http.post("http://192.168.0.16/webapp/set-template/",data,{headers : headers});
  }
  
  registerUser(user: any) : Observable<any> {
    return this.http.post("http://192.168.0.16/bsc/rest-register-user/",user);
  }

  isUsedEmailOrUser(data : any) : Observable<any> {
    return this.http.post("http://192.168.0.16/bsc/rest-user-exist/",data);
  }
}
