import { Component, OnInit } from '@angular/core';
import { LateralNavBarComponent } from '../lateral-nav-bar/lateral-nav-bar.component'
import { MatDialog, MatDialogConfig } from '@angular/material/dialog';
import { LoginService } from '../login.service';
import { Router } from '@angular/router';
import { RestService } from '../rest.service';

@Component({
  selector: 'app-main-container',
  templateUrl: './main-container.component.html',
  styleUrls: ['./main-container.component.css']
})
export class MainContainerComponent implements OnInit {

  isShowDrawler : boolean = false;
  constructor(public matDialog: MatDialog, public loginService : LoginService, public router : Router,
    public restService : RestService) {

   }

  ngOnInit(): void {
    console.log(this.loginService.hasToken());
    if(!this.loginService.hasToken()){
      this.router.navigateByUrl("/login")
    }
    this.restService.getProfileUser().subscribe( data =>{
      if(data.image == "null"){
        this.router.navigateByUrl('/biometric/enroll');
      }     
    });
  }

  showDrawler(){
    this.isShowDrawler=true;
  }

  closeDrawler(value : boolean){
    this.isShowDrawler=value;
  }
}
