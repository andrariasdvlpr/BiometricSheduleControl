import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RegisterFragmentComponent } from './register-fragment.component';

describe('RegisterFragmentComponent', () => {
  let component: RegisterFragmentComponent;
  let fixture: ComponentFixture<RegisterFragmentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ RegisterFragmentComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(RegisterFragmentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
