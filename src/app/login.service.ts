import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of} from 'rxjs';
import { CookieService} from 'ngx-cookie-service';

@Injectable({
  providedIn: 'root'
})
export class LoginService {

  constructor(private http : HttpClient, private cookies : CookieService) { }

  login(user: any) : Observable<any> {
    return this.http.post("http://192.168.0.16/api-token-auth/",user);
  }

  changePsswd(password: any) : Observable<any> {
    const token = this.getToken();
    const headers = new HttpHeaders({'authorization': 'Token '+token});
    return this.http.post("http://192.168.0.16/bsc/rest-passw-change/",password,{headers : headers});
  }

  setToken(token : string){
    this.cookies.set("token",token);
  }

  getToken(){
    return this.cookies.get("token")
  }

  deleteToken(){
    this.cookies.delete("token","/");
    this.cookies.deleteAll("/");
    this.cookies.delete("token","/app");
    this.cookies.deleteAll("/app");
  }

  hasToken() : boolean{
    return  this.cookies.check("token");
  }

}
