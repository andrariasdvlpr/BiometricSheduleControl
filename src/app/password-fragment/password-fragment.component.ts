import { Component, OnInit } from '@angular/core';
import { LoginService } from '../login.service';
import { Router} from '@angular/router';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-password-fragment',
  templateUrl: './password-fragment.component.html',
  styleUrls: ['./password-fragment.component.css']
})
export class PasswordFragmentComponent implements OnInit {

  newpsswd! : string;
  echopsswd! : string;

  constructor(public loginService : LoginService, public router : Router, public snackBar : MatSnackBar) { }

  ngOnInit(): void {
  }

  doChangePassword(){
    if(this.newpsswd === this.echopsswd){
      const password = { password : this.newpsswd};
      this.loginService.changePsswd(password).subscribe(
        res => {
        this.router.navigateByUrl('/login');
      }, err => {
        console.log(err)
        this.openSnackBar("Error en la petición");
      }
      );
    }else{
      this.openSnackBar("No coinciden las contraseñas")
    }
  }

  openSnackBar(message : string) {
    this.snackBar.open(message,'', {duration: 5000 ,panelClass:"error-snackbar"});
  }

}
