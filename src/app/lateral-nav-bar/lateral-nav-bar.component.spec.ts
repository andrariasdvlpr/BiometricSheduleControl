import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LateralNavBarComponent } from './lateral-nav-bar.component';

describe('LateralNavBarComponent', () => {
  let component: LateralNavBarComponent;
  let fixture: ComponentFixture<LateralNavBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LateralNavBarComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(LateralNavBarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
