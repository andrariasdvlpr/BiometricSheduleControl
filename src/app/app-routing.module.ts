import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LoginComponent } from './login/login.component';
import { HistoryFragmentComponent } from './history-fragment/history-fragment.component';
import { MainContainerComponent } from './main-container/main-container.component';
import { HomeFragmentComponent } from './home-fragment/home-fragment.component';
import { ProfileFragmentComponent } from './profile-fragment/profile-fragment.component';
import { CameraFragmentComponent } from './camera-fragment/camera-fragment.component';
import { PasswordFragmentComponent } from './password-fragment/password-fragment.component';
import { RegisterFragmentComponent } from './register-fragment/register-fragment.component';
import { EnrollmentFragmentComponent } from './enrollment-fragment/enrollment-fragment.component';

const routes: Routes = [
  { path: '', redirectTo: '/login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  { path: 'enroll', component : RegisterFragmentComponent },
  { path: 'biometric/enroll', component : EnrollmentFragmentComponent },
  { path: 'app', component : MainContainerComponent,
    children : [
      { path: 'home', component : HomeFragmentComponent },
      { path: 'history', component : HistoryFragmentComponent },
      { path: 'profile', component : ProfileFragmentComponent },
      { path: 'camera', component : CameraFragmentComponent },
      { path: 'psswd', component : PasswordFragmentComponent } 
    ]
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
